from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.model import GradientModel  # 또는 사용자 정의 모델 클래스
from functorch import make_functional, jacrev, vmap
import scipy
from scipy import linalg
import contextlib
from joblib import Parallel, delayed
import joblib
from tqdm import tqdm
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """tqdm progress bar와 joblib를 연동하는 context manager"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
############################################
# model deviation 관련 유틸리티 함수들
############################################
def empirical_ntk_jacobian_contraction(fnet, params, x1, x2, batch_size=128, cpu=False, regression=False, return_features=False):
    """
    NTK 추정을 위해 각 입력의 jacobian을 계산하고, 두 입력 간 내적을 통해 kernel matrix를 산출합니다.
    """
    def fnet_single(params, x):
        # regression 여부와 상관없이 전체 출력 벡터를 반환하도록 변경
        return fnet(params, x.unsqueeze(0)).squeeze(0)

    
    def get_jac(x, cpu=False):
        def get_batch_jac(x_batch):
            jac = vmap(jacrev(fnet_single), (None, 0))(params, x_batch)
            jac = [j.flatten(1) for j in jac]
            jac = torch.hstack(jac)
            return jac
        num_dp = x.shape[0]
        batch_num = num_dp // batch_size
        residue = num_dp % batch_size
        all_jac = []
        en = 0
        for idx in range(batch_num):
            st = idx * batch_size
            en = (idx + 1) * batch_size
            jac = get_batch_jac(x[st:en])
            if cpu:
                jac = jac.cpu()
            all_jac.append(jac)
        if residue:
            jac = get_batch_jac(x[en:])
            if cpu:
                jac = jac.cpu()
            all_jac.append(jac)
        all_jac = torch.vstack(all_jac)
        return all_jac

    same_data = x1.data_ptr() == x2.data_ptr() and x1.shape[0] == x2.shape[0]
    if same_data:
        all_jac = get_jac(x1, cpu=cpu)
        result = all_jac @ all_jac.transpose(0, 1)
        if not return_features:
            del all_jac
    else:
        all_jac1 = get_jac(x1, cpu=cpu)
        all_jac2 = get_jac(x2, cpu=cpu)
        result = all_jac1 @ all_jac2.transpose(0, 1)
        del all_jac1, all_jac2
    if return_features:
        return result.cpu().detach().numpy(), all_jac.cpu().detach().numpy()
    else:
        return result.cpu().detach().numpy()

def linear_solver_regression(A, b, mu=0):
    """
    릿지 정규화(mu)를 적용한 선형 시스템의 해를 구합니다.
    A와 b는 numpy 배열이어야 합니다.
    """
    dim_A = A.shape[0]
    ridge_A = A + mu * np.eye(dim_A)
    inv_A = linalg.pinv(ridge_A)
    result = np.matmul(inv_A, b)
    return result
def compute_score(alpha, beta, kernel_matrix_full, kernel_matrix_exclude, kernel_matrix_cross, regression=True):
    if regression:
        score_0 = np.matmul(np.matmul(alpha.T, kernel_matrix_full), alpha)
        score_1 = np.matmul(np.matmul(beta.T, kernel_matrix_exclude), beta)
        score_2 = np.matmul(np.matmul(alpha.T, kernel_matrix_cross), beta)
    # 각 term의 대각합(trace)를 구해 스칼라로 축소합니다.
    score_scalar = np.trace(score_0) + np.trace(score_1) - 2 * np.trace(score_2)
    return score_scalar
def compute_deviation_for_one(i, kernel_matrix, y_train_onehot, alpha, mu):
    """
    i번째 데이터를 제외한 leave-one-out 방식으로 deviation score를 계산하는 함수.
    
    Parameters:
      - i: int, 제외할 인덱스
      - kernel_matrix: np.ndarray, 전체 kernel matrix
      - y_train_onehot: np.ndarray, one-hot 인코딩된 라벨 배열
      - alpha: np.ndarray, 전체 데이터에 대한 해
      - mu: float, 정규화 상수
    """
    n = kernel_matrix.shape[0]
    indices = list(range(n))
    indices.remove(i)
    K_exclude = kernel_matrix[np.ix_(indices, indices)]
    K_cross = kernel_matrix[:, indices]
    y_exclude = y_train_onehot[indices]
    beta = linear_solver_regression(K_exclude, y_exclude, mu=mu)
    score = compute_score(alpha, beta, kernel_matrix, K_exclude, K_cross, regression=True)
    return score

############################################
# opendataval 기반 Model Deviation Evaluator
############################################
class ModelDeviationEvaluator(DataEvaluator, ModelMixin):
    """
    Model deviation을 활용한 data valuation evaluator.
    
    모델을 학습한 후, functorch를 통해 NTK 기반 kernel matrix를 계산하고,
    전체 데이터(α 해)와 leave-one-out 방식(β 해)으로 모델의 deviation score를 산출합니다.
    
    Parameters
    ----------
    mu : float, optional
        릿지 정규화 상수 (기본값: 1e-3)
    num_classes : int, optional
        분류 문제의 클래스 수 (기본값: 10)
    """
    def __init__(self, mu: float = 1e-3, num_classes: int = 10):
        self.mu = mu
        self.num_classes = num_classes

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor = None,
        y_valid: torch.Tensor = None,
    ):
        """
        데이터를 입력받아 evaluator 내부에 저장합니다.
        x_train과 y_train은 필수이며, x_valid, y_valid는 옵션입니다.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        return self

    def input_model(self, pred_model: GradientModel):
        """
        예측 모델을 입력받아 evaluator 내부에 저장합니다.
        모델은 clone() 메서드를 지원해야 합니다.
        """
        assert (
            hasattr(pred_model, "clone") or callable(getattr(pred_model, "forward", None))
        ), "A cloneable model or a forward method is required."
        self.pred_model = pred_model.clone()
        return self

    def train_data_values(self, *args, **kwargs):
        """
        예측 모델을 x_train, y_train에 맞춰 학습시키고,
        NTK 기반 kernel matrix 및 선형 시스템 해(α)를 계산합니다.
        """
        self.pred_model.fit(self.x_train, self.y_train, *args, **kwargs)

        # 모델을 통해 x_train에 대한 임베딩을 계산합니다.
        # with torch.no_grad():
        #     # 원본 입력(self.x_train)을 그대로 사용합니다.
        #     _ = self.pred_model(self.x_train)
        # functorch를 통해 모델을 함수형으로 변환
        fnet, params = make_functional(self.pred_model)
        def ntk_kernel(x1, x2, mode='ntk'):
            return empirical_ntk_jacobian_contraction(fnet, params, x1, x2, cpu=True, regression=True)
        # 원본 입력(self.x_train)을 사용하여 kernel matrix 계산
        self.kernel_matrix = ntk_kernel(self.x_train, self.x_train, mode='ntk')
        
        # y_train을 one-hot 인코딩 (이미 one-hot이면 그대로 사용)
        if self.y_train.dim() == 1 or self.y_train.size(1) == 1:
            self.y_train_onehot = torch.nn.functional.one_hot(self.y_train.long(), num_classes=self.num_classes).float()
        else:
            self.y_train_onehot = self.y_train.float()
            
        # 선형 시스템 (K + mu I) α = y_train_onehot 을 풀어 α 해 계산 (numpy 배열로 변환)
        self.alpha = linear_solver_regression(self.kernel_matrix, self.y_train_onehot.cpu().numpy(), mu=self.mu)
        return self

    def evaluate_data_values(self, thread_num: int = 4) -> np.ndarray:
        """
        각 training 데이터 포인트에 대해 leave-one-out 방식으로 deviation score를 병렬로 계산합니다.
        """
        n = self.x_train.shape[0]
        # 필요한 변수들을 pickle 가능한 객체로 준비합니다.
        kernel_matrix = self.kernel_matrix  # numpy array
        y_train_onehot = self.y_train_onehot.cpu().numpy()  # numpy array
        alpha = self.alpha  # numpy array
        mu = self.mu
        
        with tqdm_joblib(tqdm(desc="Computing deviation scores", total=n)) as progress_bar:
            deviation_scores = Parallel(n_jobs=20, max_nbytes=5000)(
                delayed(compute_deviation_for_one)(i, kernel_matrix, y_train_onehot, alpha, mu) for i in range(n)
            )
        return np.array(deviation_scores)
