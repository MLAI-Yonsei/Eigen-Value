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
    AME,
    DVRL,
    BetaShapley,
    DataBanzhaf,
    DataOob,
    DataShapley,
    InfluenceSubsample,
    KNNShapley,
    LavaEvaluator,
    LeaveOneOut,
    RandomEvaluator,
    RobustVolumeShapley,
    SingularLavaEvaluator,
    singularKNNShapley,
    ModelDeviationEvaluator,
    singularDataOob
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

from torchvision.models.resnet import ResNet50_Weights, resnet50
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

    # not use
    parser.add_argument('--data_i', type=int, default=0)    

    args = parser.parse_args()

    return args

def main(i=0):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    
    dataset_name = "random_dataset"
    noise_rate = 0.1

    embedder = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    embedder.fc = nn.Identity()

    
    size = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # CIFAR-10 
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    
    num_samples = 3000
    test_samples = 1000
    set_seed()
    random_train_indices = random.sample(range(len(trainset)), num_samples)
    random_test_indices = random.sample(range(len(testset)), test_samples)

    train_subset = torch.utils.data.Subset(trainset, random_train_indices)
    test_subset = torch.utils.data.Subset(testset, random_test_indices)

    
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

    embeddings_train = []
    labels_train = []
    with torch.no_grad():
        for images, target in trainloader:
            images = images.to(device)
            target = target.to(device)
            outputs = embedder(images)
            embeddings_train.append(outputs)
            labels_train.append(target)

    embeddings_test = []
    labels_test = []
    with torch.no_grad():
        for images, target in testloader:
            images = images.to(device)
            target = target.to(device)
            outputs = embedder(images)
            embeddings_test.append(outputs)
            labels_test.append(target)

    # Convert the list of tensors to a single tensor
    embeddings_train = torch.cat(embeddings_train)
    labels_train = torch.cat(labels_train)
    embeddings_test = torch.cat(embeddings_test)
    labels_test = torch.cat(labels_test)

    # Combine train and test embeddings and labels
    X_combined = torch.cat((embeddings_train, embeddings_test), dim=0)
    Y_combined = torch.cat((labels_train, labels_test), dim=0)






    X_combined = X_combined.cpu()
    Y_combined = Y_combined.cpu()

    # Register a dataset from arrays X and y

    pd_dataset = Register(dataset_name=dataset_name, one_hot=True).from_data(X_combined.numpy(), Y_combined.numpy())


    
    values = random_train_indices
    
    train_random_np =np.arange(0, 2000)# np.array(train_random)
    validation_remain_np = np.arange(2000, 3000)#np.array(validation_remain)
    valid_count =1000# 2000 - int(num_samples * 0.9)
    test_np = np.arange(3000,4000)#np.array(random_test_indices)


    fetcher = (
        DataFetcher(dataset_name, '../data_files/', False)
        .split_dataset_by_indices(train_indices=train_random_np,
                                valid_indices=validation_remain_np,
                                test_indices=test_np)
        .noisify(mix_labels, noise_rate=noise_rate)
    )


    set_seed()
    pred_model = LogisticRegression(fetcher.covar_dim[0], num_classes = 10)
        
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    train_kwargs = {"epochs": 3, "batch_size": 100, "lr": 0.01}
    exper_med = ExperimentMediator(fetcher, pred_model, train_kwargs = train_kwargs, metric_name = "accuracy")
    

    data_evaluators = [ 
        
        RandomEvaluator(),
        SingularLavaEvaluator(),
        singularKNNShapley(k_neighbors=valid_count),        
        InfluenceSubsample(num_models=1000), # influence function        
        KNNShapley(k_neighbors=valid_count), # KNN-Shapley                
        DataOob(num_models=800), # Data-OOB
        LavaEvaluator(), # LAVA
        ModelDeviationEvaluator(),
        singularDataOob(num_models=800),
        
    ]


    exper_med = exper_med.compute_data_values(data_evaluators=data_evaluators)
    
    output_dir = "./datavalues/"
    exper_med.set_output_directory(output_dir)
    from opendataval.experiment.exper_methods import save_dataval
    exper_med.evaluate(save_dataval, save_output=True)
    
    
    
    





    from opendataval.experiment.exper_methods import save_dataval



if __name__ == "__main__":

    running_start = time.time()
    main()
    running_end = time.time()
    full_running_time = running_end - running_start
    print(f"Full Running Time: ", full_running_time)
