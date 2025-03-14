from typing import Optional

import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader
import math
from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.model.api import Model


class weightsingularKNNShapley(DataEvaluator, ModelLessMixin):
    """Data valuation using KNNShapley implementation.

    KNN Shapley is a model-less mixin. This means we cannot specify an underlying
    prediction model for the DataEvaluator. However, we can specify a pretrained
    embedding model.

    References
    ----------
    .. [1] R. Jia et al.,
        Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1908.08619.

    Parameters
    ----------
    k_neighbors : int, optional
        Number of neighbors to group the data points, by default 10
    batch_size : int, optional
        Batch size of tensors to load at a time during training, by default 32
    embedding_model : Model, optional
        Pre-trained embedding model used by DataEvaluator, by default None
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        k_neighbors: int = 10,
        batch_size: int = 32,
        embedding_model: Optional[Model] = None,
        random_state: Optional[RandomState] = None,
        weight : int = 1,
        epsilon_weight : int = 1
    ):
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.random_state = check_random_state(random_state)
        self.weight = weight
        self.epsilon_weight = epsilon_weight
        

    def match(self, y: torch.Tensor) -> torch.Tensor:
        """:math:`1.` for all matching rows and :math:`0.` otherwise."""
        return (y == self.y_valid).all(dim=1).float()

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Computes KNN shapley data values, as implemented by the following. Ignores all
        positional and key word arguments.

        References
        ----------
        .. [1] PyTorch implementation
            <https://github.com/AI-secure/Shapley-Study/blob/master/shapley/measures/KNN_Shapley.py>
        """
        n = len(self.x_train)
        m = len(self.x_valid)
        self.d = self.x_train.shape[1]  # 입력 feature의 차원
        x_train, x_valid = self.embeddings(self.x_train, self.x_valid)

        # Computes Euclidean distance by computing crosswise per batch
        # Doesn't shuffle to maintain relative order
        x_train_view, x_valid_view = x_train.view(n, -1), x_valid.view(m, -1)

        dist_list = []  # Uses batching to only load at most `batch_size` tensors
        for x_train_batch in DataLoader(x_train_view, self.batch_size):  # No shuffle
            dist_row = []
            for x_val_batch in DataLoader(x_valid_view, self.batch_size):
                dist_row.append(torch.cdist(x_train_batch, x_val_batch))
            dist_list.append(torch.cat(dist_row, dim=1))
        dist = torch.cat(dist_list, dim=0)

        # Arranges by distances
        sort_indices = torch.argsort(dist, dim=0, stable=True)
        y_train_sort = self.y_train[sort_indices]

        score = torch.zeros_like(dist)
        score[sort_indices[n - 1], range(m)] = self.match(y_train_sort[n - 1]) / n

        # fmt: off
        for i in tqdm.tqdm(range(n - 2, -1, -1)):
            score[sort_indices[i], range(m)] = (
                score[sort_indices[i + 1], range(m)]
                + min(self.k_neighbors, i + 1) / (self.k_neighbors * (i + 1))
                * (self.match(y_train_sort[i]) - self.match(y_train_sort[i + 1]))
            )
                # === 추가 알고리즘: 각 training sample의 influence 계산 ===
        
        # 1. 데이터 정규화
        x_mean = torch.mean(self.x_train, dim=0, keepdim=True)          # (1, d)
        x_std = torch.std(self.x_train, dim=0, keepdim=True) + 1e-6       # (1, d)
        x_norm = (self.x_train - x_mean) / x_std                         # (n, d)

        # 2. 공분산 행렬 Σ 계산: (d, d)
        Sigma = (x_norm.t() @ x_norm) / n

        # 3. SVD를 통한 singular value 및 vector 산출
        U, S, _ = torch.linalg.svd(Sigma)
        sigma_max = S[0]            # 최대 singular value
        sigma_min = S[-1]           # 최소 singular value
        u_max = U[:, 0]             # (d,)
        u_min = U[:, -1]            # (d,)

        # 4. 각 샘플에 대한 u_max, u_min 상의 투영값 계산
        proj_max = torch.matmul(x_norm, u_max)  # (n,)
        proj_min = torch.matmul(x_norm, u_min)  # (n,)

        # 5. 각 샘플의 δ 값 계산: δ_max, δ_min
        # delta_max =  -(proj_max ** 2) / self.n    # (n,)
        delta_max = -(proj_max ** 2) / n + torch.tensor(sigma_max, device=sigma_max.device)/ n * self.weight# (n,)
        delta_min =  -(proj_min ** 2) / n    # (n,)
        # delta_min = -(proj_min ** 2) / self.n + torch.tensor(sigma_min, device=sigma_min.device)  # (n,)

        # 6. 상수 항 계산
        d_val = float(self.d)
        factor1 = (sigma_max * math.sqrt(d_val) + math.sqrt(d_val**2 - d_val)) / (sigma_min ** 2)
        factor2 = math.sqrt(d_val) / sigma_min

        # 7. 최종 influence tensor 계산
        scaling_factor_1 =  1 / math.sqrt(self.d ** 2 - self.d) 
        singular_value_term = (-factor1 * delta_min + factor2 * delta_max) * scaling_factor_1  # (n,)

        # === 두 값 결합 ===
        # 최종 데이터 값은 LAVA gradient와 influence를 합산한 값으로 결정
        # final_data_values = train_gradient + singular_value_term
        scaling_factor_2 = score.mean(axis=1).abs().mean() / singular_value_term.abs().mean() * self.epsilon_weight
        self.data_values = (score.mean(axis=1) + singular_value_term * scaling_factor_2).detach().numpy()
        # self.data_values = score.mean(axis=1).detach().numpy() + singular_value_term.detach().numpy()

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Compute data values using KNN Shapley data valuation

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """
        return self.data_values
