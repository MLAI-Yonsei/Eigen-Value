from typing import Optional
import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state
import math

from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.dataval.lava.otdd import DatasetDistance, FeatureCost
from opendataval.model import Model

def macos_fix():
    """Geomloss package has a bug on MacOS remedied as follows.
    Link to similar bug: https://github.com/NVlabs/stylegan3/issues/75
    """
    import os
    import sys
    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class SingularLavaEvaluator(DataEvaluator, ModelLessMixin):
    """Data valuation using LAVA implementation with additional influence calculation.

    References
    ----------
    [1] H. A. Just et al., LAVA: Data Valuation without Pre-Specified Learning Algorithms, 2023.
    [2] D. Alvarez-Melis and N. Fusi, Geometric Dataset Distances via Optimal Transport, 2020.
    [3] D. Alvarez-Melis and N. Fusi, Dataset Dynamics via Gradient Flows in Probability Space, 2020.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        embedding_model: Optional[Model] = None,
        random_state: Optional[RandomState] = None,
    ):
        macos_fix()
        torch.manual_seed(check_random_state(random_state).tomaxint())
        self.embedding_model = embedding_model
        self.device = device
        
        

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Computes the class-wise Wasserstein distance between the training and the
        validation set.
        """
        feature_cost = None
        if hasattr(self, "embedding_model") and self.embedding_model is not None:
            resize = 32
            feature_cost = FeatureCost(
                src_embedding=self.embedding_model,
                src_dim=(3, resize, resize),
                tgt_embedding=self.embedding_model,
                tgt_dim=(3, resize, resize),
                p=2,
                device=self.device.type,
            )
        self.n = self.x_train.shape[0]  
        self.d = self.x_train.shape[1]  
        x_train, x_valid = self.embeddings(self.x_train, self.x_valid)
        dist = DatasetDistance(
            x_train=x_train,
            y_train=self.y_train,
            x_valid=x_valid,
            y_valid=self.y_valid,
            feature_cost=feature_cost if feature_cost else "euclidean",
            lam_x=1.0,
            lam_y=1.0,
            p=2,
            entreg=1e-1,
            device=self.device,
        )
        self.dual_sol = dist.dual_sol()
        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Combines the calibrated dual solution gradient with an additional
        influence measure computed from the training data.

        Returns
        -------
        np.ndarray
            Final data values for each training input data point.
        """
        
        f1k = self.dual_sol[0].squeeze()  
        num_points = f1k.shape[0] - 1   
        train_gradient = f1k * (1 + 1 / num_points) - f1k.sum() / num_points
        
        train_gradient = -1 * train_gradient

        
        x_mean = torch.mean(self.x_train, dim=0, keepdim=True)          # (1, d)
        x_std = torch.std(self.x_train, dim=0, keepdim=True) + 1e-6       # (1, d)
        x_norm = (self.x_train - x_mean) / x_std                         # (n, d)

        
        Sigma = (x_norm.t() @ x_norm) / self.n

        
        U, S, _ = torch.linalg.svd(Sigma)
        sigma_max = S[0]            
        sigma_min = S[-1]           
        u_max = U[:, 0]             
        u_min = U[:, -1]            

        
        proj_max = torch.matmul(x_norm, u_max)  # (n,)
        proj_min = torch.matmul(x_norm, u_min)  # (n,)

        
        # delta_max =  -(proj_max ** 2) / self.n    # (n,)
        delta_max = -(proj_max ** 2) / self.n + torch.tensor(sigma_max, device=sigma_max.device)/self.n
        # delta_min =  -(proj_min ** 2) / self.n    # (n,)
        delta_min = -(proj_min ** 2) / self.n #+ torch.tensor(sigma_min, device=sigma_min.device)  # (n,)

        
        d_val = float(self.d)
        factor1 = (sigma_max * math.sqrt(d_val) + math.sqrt(d_val**2 - d_val)) / (sigma_min ** 2)
        factor2 = math.sqrt(d_val) / sigma_min

        
        scaling_factor =  1 / math.sqrt(self.d ** 2 - self.d)
        singular_value_term = (-factor1 * delta_min + factor2 * delta_max) * scaling_factor  # (n,)

        
        final_data_values = train_gradient + singular_value_term

        
        return final_data_values.cpu().numpy()