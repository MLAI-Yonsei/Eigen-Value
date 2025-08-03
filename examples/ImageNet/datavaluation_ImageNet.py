import matplotlib.pyplot as plt
# Imports
import numpy as np
import pandas as pd
import torch.nn.functional as F
import time

import torch
import torchvision
import torchvision.transforms as transforms
import random
import argparse
# Opendataval
from opendataval.dataloader import Register, DataFetcher, mix_labels, add_gauss_noise
from opendataval.dataval import (
    RandomEvaluator,
    SingularLavaEvaluator,
    singularKNNShapley,
    InfluenceSubsample,
    KNNShapley,
    DataOob,
    LavaEvaluator,
    singularDataOob,
    ModelDeviationEvaluator,
    weightsingularKNNShapley,
    weightsingularDataOob,
    weightSingularLavaEvaluator
)
from opendataval.experiment.exper_methods import (
    discover_corrupted_sample,
    noisy_detection,
    remove_high_low,
    save_dataval
)

from opendataval.experiment import ExperimentMediator
from opendataval.model.lenet import LeNet
from opendataval.model.api import ClassifierSkLearnWrapper,ClassifierUnweightedSkLearnWrapper

from sklearn.ensemble import RandomForestClassifier

from torchvision.models.resnet import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from torchvision.models import vit_b_16, vit_l_16, ViT_B_16_Weights, ViT_L_16_Weights
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from opendataval.model.mlp import ClassifierMLP, RegressionMLP
from opendataval.model.logistic_regression import LogisticRegression

import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', 
                      choices=['resnet18', 'resnet50', 'vit_b_16', 'vit_l_16'],
                      help='Model to use for feature extraction')
    parser.add_argument('--embedding_dir', type=str, help='Directory containing domain embeddings and labels')
    parser.add_argument('--output_dir', type=str,  help='Directory to save datavaluation results')
    return parser.parse_args()

def get_embedder(model_name, device):
    if model_name == 'resnet18':
        embedder = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
        embedder.fc = nn.Identity()
    elif model_name == 'resnet50':
        embedder = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        embedder.fc = nn.Identity()
    elif model_name == 'vit_b_16':
        embedder = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device)
        embedder.heads = nn.Identity()
    elif model_name == 'vit_l_16':
        embedder = vit_l_16(weights=ViT_L_16_Weights.DEFAULT).to(device)
        embedder.heads = nn.Identity()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    embedder.eval()
    return embedder

def main():
    args = create_parse()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    dataset_name = "random_dataset"
    noise_rate = 0.1

    

    embeddings = torch.load(args.embedding_dir + '/embeddings.pt', map_location=device)
    labels = torch.load(args.embedding_dir + '/labels.pt', map_location=device)

    X_combined = embeddings.cpu()
    Y_combined = labels.cpu()

    # Register a dataset from arrays X and y
    pd_dataset = Register(dataset_name=dataset_name, one_hot=True).from_data(X_combined.numpy(), Y_combined.numpy())
    train_size = 30000
    valid_size = 1000
    test_size  = 9000
    train_random_np = np.arange(0, train_size)
    validation_remain_np = np.arange(train_size, train_size+valid_size)
    test_np = np.arange(train_size+valid_size, train_size+valid_size+test_size)

    fetcher = (
        DataFetcher(dataset_name, '../data_files/', False)
        .split_dataset_by_indices(train_indices=train_random_np,
                                valid_indices=validation_remain_np,
                                test_indices=test_np)
        .noisify(mix_labels, noise_rate=noise_rate)
    )

    set_seed()
    pred_model = LogisticRegression(fetcher.covar_dim[0], num_classes=1000)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    train_kwargs = {"epochs": 3, "batch_size": 100, "lr": 0.001}
    exper_med = ExperimentMediator(fetcher, pred_model, train_kwargs=train_kwargs, metric_name="accuracy")

    data_evaluators = [ 
        RandomEvaluator(),
        # SingularLavaEvaluator(),
        singularKNNShapley(k_neighbors=valid_size),        
        InfluenceSubsample(num_models=1000), # influence function        
        KNNShapley(k_neighbors=valid_size), # KNN-Shapley                
        DataOob(num_models=800), # Data-OOB
        # LavaEvaluator(), # LAVA
        # ModelDeviationEvaluator(),
        singularDataOob(num_models=800),
        # weightSingularLavaEvaluator(weight = 1, epsilon_weight = 1),
        # weightSingularLavaEvaluator(weight = 1, epsilon_weight = -1),
        # weightSingularLavaEvaluator(weight = 1, epsilon_weight = 0.1),
        # weightSingularLavaEvaluator(weight = 1, epsilon_weight = -0.1),
        # weightSingularLavaEvaluator(weight = 1, epsilon_weight = 0.01),
        # weightSingularLavaEvaluator(weight = 1, epsilon_weight = -0.01),
        # weightSingularLavaEvaluator(weight = 1, epsilon_weight = 0.001),
        # weightSingularLavaEvaluator(weight = 1, epsilon_weight = -0.001),
        # weightSingularLavaEvaluator(weight = 0, epsilon_weight = 1),
        # weightSingularLavaEvaluator(weight = 0, epsilon_weight = -1),
        # weightSingularLavaEvaluator(weight = 0, epsilon_weight = 0.1),
        # weightSingularLavaEvaluator(weight = 0, epsilon_weight = -0.1),
        # weightSingularLavaEvaluator(weight = 0, epsilon_weight = 0.01),
        # weightSingularLavaEvaluator(weight = 0, epsilon_weight = -0.01),
        # weightSingularLavaEvaluator(weight = 0, epsilon_weight = 0.001),
        # weightSingularLavaEvaluator(weight = 0, epsilon_weight = -0.001),
        
        # weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = 1),
        # weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = -1),
        # weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = 0.1),
        # weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = -0.1),
        weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = 0.01),
        weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = -0.01),
        weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = 0.001),
        weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = -0.001),
        # weightsingularDataOob(num_models=800, weight = 0, epsilon_weight = 1),
        # weightsingularDataOob(num_models=800, weight = 0, epsilon_weight = -1),
        # weightsingularDataOob(num_models=800, weight = 0, epsilon_weight = 0.1),
        # weightsingularDataOob(num_models=800, weight = 0, epsilon_weight = -0.1),
        weightsingularDataOob(num_models=800, weight = 0, epsilon_weight = 0.01),
        weightsingularDataOob(num_models=800, weight = 0, epsilon_weight = -0.01),
        weightsingularDataOob(num_models=800, weight = 0, epsilon_weight = 0.001),
        weightsingularDataOob(num_models=800, weight = 0, epsilon_weight = -0.001),
        weightsingularKNNShapley(k_neighbors=valid_size, weight = 0, epsilon_weight = 0.1),
        weightsingularKNNShapley(k_neighbors=valid_size, weight = 0, epsilon_weight = -0.1),
        weightsingularKNNShapley(k_neighbors=valid_size, weight = 0, epsilon_weight = 0.01),
        weightsingularKNNShapley(k_neighbors=valid_size, weight = 0, epsilon_weight = -0.01),
        weightsingularKNNShapley(k_neighbors=valid_size, weight = 0, epsilon_weight = 0.001),
        weightsingularKNNShapley(k_neighbors=valid_size, weight = 0, epsilon_weight = -0.001),
        # weightsingularKNNShapley(k_neighbors=valid_size, weight = 1, epsilon_weight = 0.1),
        # weightsingularKNNShapley(k_neighbors=valid_size, weight = 1, epsilon_weight = -0.1),
        # weightsingularKNNShapley(k_neighbors=valid_size, weight = 1, epsilon_weight = 0.01),
        # weightsingularKNNShapley(k_neighbors=valid_size, weight = 1, epsilon_weight = -0.01),
        # weightsingularKNNShapley(k_neighbors=valid_size, weight = 1, epsilon_weight = 0.001),
        # weightsingularKNNShapley(k_neighbors=valid_size, weight = 1, epsilon_weight = -0.001),
        # ModelDeviationEvaluator()
    ]

    exper_med = exper_med.compute_data_values(data_evaluators=data_evaluators)
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    exper_med.set_output_directory(output_dir)
    exper_med.evaluate(save_dataval, save_output=True)

if __name__ == "__main__":
    running_start = time.time()
    main()
    running_end = time.time()
    full_running_time = running_end - running_start
    print(f"Full Running Time: ", full_running_time)
