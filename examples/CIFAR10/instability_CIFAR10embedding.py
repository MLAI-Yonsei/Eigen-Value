import matplotlib.pyplot as plt
# Imports
import numpy as np
import pandas as pd
import torch.nn.functional as F
import time
# import tensorflow as tf
import torch
import torchvision
import torchvision.transforms as transforms
import random
import argparse
import time
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



def main(data_i=0):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Set up hyperparameters
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


    if ver_ID:
        noise_transform = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    else:
        noise_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    
    num_samples = 300-data_i
    test_samples = 100
    set_seed()
    random_train_indices = random.sample(range(len(trainset)), num_samples)
    random_test_indices = random.sample(range(len(testset)), test_samples)
    remain_indices = list(set(range(len(trainset))) - set(random_train_indices))
    
    
    seed = int(time.time()) % 60  
    set_seed(seed)
    selected_noise_indices = random.sample(remain_indices, data_i)

    

    train_subset = torch.utils.data.Subset(trainset, random_train_indices)
    test_subset = torch.utils.data.Subset(testset, random_test_indices)

    
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)
    noise_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=noise_transform)
    
    noise_subset = torch.utils.data.Subset(noise_trainset, selected_noise_indices)
    noise_loader = torch.utils.data.DataLoader(noise_subset, batch_size=64, shuffle=False)

    
    embedder = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    embedder.fc = nn.Identity()

    
    embeddings_train = []
    labels_train = []
    with torch.no_grad():
        for images, target in trainloader:
            images = images.to(device)
            target = target.to(device)
            outputs = embedder(images)
            embeddings_train.append(outputs)
            labels_train.append(target)
    
    with torch.no_grad():
        for images, target in noise_loader:
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

    
    embeddings_train = torch.cat(embeddings_train)
    labels_train = torch.cat(labels_train)
    embeddings_test = torch.cat(embeddings_test)
    labels_test = torch.cat(labels_test)

    X_combined = torch.cat((embeddings_train, embeddings_test), dim=0)
    Y_combined = torch.cat((labels_train, labels_test), dim=0)






    X_combined = X_combined.cpu()
    Y_combined = Y_combined.cpu()

    # Register a dataset from arrays X and y

    pd_dataset = Register(dataset_name=dataset_name, one_hot=True).from_data(X_combined.numpy(), Y_combined.numpy())


    
    # values = random_train_indices
    
    train_random_np = np.arange(100,300)
    
    validation_remain_np = np.arange(0, 100)#np.array(validation_remain)
    valid_count =1000# 2000 - int(num_samples * 0.9)
    test_np = np.arange(300,400)#np.array(random_test_indices)


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

    set_seed(seed)
    exper_med = exper_med.compute_data_values(data_evaluators=data_evaluators)
    
    if ver_ID:
        dir_ID = "TRUE"
    else:
        dir_ID = "FALSE"
    output_dir = f"./instability/{dir_ID}/{seed}"
    exper_med.set_output_directory(output_dir)
    from opendataval.experiment.exper_methods import save_dataval
    exper_med.evaluate(save_dataval, save_output=True)
    
   




    from opendataval.experiment.exper_methods import save_dataval




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--data_i', type=int, default=0)
    parser.add_argument('--ver_ID', action='store_true', default=False)  

    cmd_args = vars(parser.parse_args())
    data_i = cmd_args['data_i']
    ver_ID = cmd_args['ver_ID']


    running_start = time.time()
    main(data_i)
    running_end = time.time()
    full_running_time = running_end - running_start
    print(f"Full Running Time: ", full_running_time)
