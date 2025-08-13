import numpy as np
import pandas as pd
import torch
import random
import os
import torch.nn as nn
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from PIL import Image as PILImage
import torchvision.transforms as transforms
import torchvision
import argparse
import torch.optim as optim
import torch.nn.functional as F

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
)
from opendataval.model.logistic_regression import LogisticRegression
from opendataval.experiment import ExperimentMediator
from torchvision.models.resnet import ResNet50_Weights, resnet50


device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
IMG_SIZE = (224, 224)
transform = transforms.Compose([
    transforms.Resize(size=IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed()

def get_embedder():
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    model.fc = nn.Identity()  
    model.eval()
    return model


embedder = get_embedder()


def return_indices(model, df):
    return df[df['Unnamed: 0'] == f"{model}"]['indices'].tolist()


def make_trainloader(model, dataval_df, num=0):
    set_seed()

    random_1000_idx = random.sample(list(dataval_df['indices'].unique()), 1000)
        
    filtered_df = dataval_df[dataval_df['indices'].isin(random_1000_idx)]

    grouped_df = filtered_df.sort_values(by=['Unnamed: 0', 'data_values'], ascending=ascending).groupby('Unnamed: 0') \
                .head(num) \
                [['Unnamed: 0','indices']]
    sorted_df = grouped_df.copy()
    
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    num_samples = 4000
    set_seed()
    random_train_indices = random.sample(range(len(trainset)), num_samples)
    
    
    indices = random_1000_idx.copy() if num == len(random_1000_idx) else return_indices(model=f'{model}', df=sorted_df)
    
    wholeindex = dataval_df['indices'].unique()
    valid_indices = list(set(range(num_samples)) - set(wholeindex))
    
    random_trainset = torch.utils.data.Subset(trainset, random_train_indices)
    selected_data = torch.utils.data.Subset(random_trainset, indices)
    unselected_data = torch.utils.data.Subset(trainset, valid_indices)
    
    train_loader = torch.utils.data.DataLoader(selected_data, batch_size=128, shuffle=True, num_workers=0)
    # validation_loader = torch.utils.data.DataLoader(unselected_data, batch_size=128, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
    
    
    def compute_embeddings(loader):
        embeddings, labels = [], []
        with torch.no_grad():
            for images, target in loader:
                images = images.to(device)
                target = target.to(device)
                outputs = embedder(images)
                embeddings.append(outputs.view(outputs.size(0), -1))
                labels.append(target)
        return torch.cat(embeddings), torch.cat(labels)
    
    embeddings_train, labels_train = compute_embeddings(train_loader)
    
    embeddings_test, labels_test = compute_embeddings(test_loader)
    
    
    train_dataset = torch.utils.data.TensorDataset(embeddings_train, labels_train)
    
    test_dataset  = torch.utils.data.TensorDataset(embeddings_test, labels_test)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return train_dataloader, test_dataloader
def compute_embeddings(loader):
    embeddings, labels = [], []
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            target = target.to(device)
            outputs = embedder(images)
            embeddings.append(outputs.view(outputs.size(0), -1))
            labels.append(target)
    return torch.cat(embeddings), torch.cat(labels)

def prepare_full_dataloaders():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    full_train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    full_test_loader  = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
    
    def compute_embeddings(loader):
        embeddings, labels = [], []
        with torch.no_grad():
            for images, target in loader:
                images = images.to(device)
                target = target.to(device)
                outputs = embedder(images)
                embeddings.append(outputs.view(outputs.size(0), -1))
                labels.append(target)
        return torch.cat(embeddings), torch.cat(labels)
    
    embeddings_train, labels_train = compute_embeddings(full_train_loader)
    embeddings_test, labels_test = compute_embeddings(full_test_loader)
    
    train_dataset = torch.utils.data.TensorDataset(embeddings_train, labels_train)
    test_dataset  = torch.utils.data.TensorDataset(embeddings_test, labels_test)
    
    full_train_loader_emb = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
    full_test_loader_emb  = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return full_train_loader_emb, full_test_loader_emb, embeddings_train.shape[1]


def train_full_model(train_loader, test_loader, modelname, input_dim):
    model = LogisticRegression(input_dim=input_dim, num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        correct_train, total_train = 0, 0
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        train_accuracy = 100 * correct_train / total_train
    print(f'Methodology : {modelname}, Final Train Accuracy: {train_accuracy:.2f}%')
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    test_accuracy = 100 * correct / total
    print(f'Methodology : {modelname}, Final Test Accuracy: {test_accuracy:.2f}%')
    return model, train_accuracy, test_accuracy

def train_model(train_loader, validation_loader, test_loader, modelname, input_dim):
    model = LogisticRegression(input_dim=input_dim, num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        correct_train, total_train = 0, 0
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        train_accuracy = 100 * correct_train / total_train
        
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for data, target in validation_loader:
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()
        val_loss /= len(validation_loader)
        val_accuracy = 100 * correct_val / total_val
    print(f'Methodology : {modelname}, Final Train Accuracy: {train_accuracy:.2f}%, '
          f'Final Validation Loss: {val_loss:.4f}, Final Validation Accuracy: {val_accuracy:.2f}%')
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    test_accuracy = 100 * correct / total
    print(f'Methodology : {modelname}, Final Test Accuracy: {test_accuracy:.2f}%')
    return model, train_accuracy, val_accuracy, test_accuracy

# ====================================================
# CIFAR-10C (corrupted data) 
# ====================================================
def get_file_prefixes(directory_path):
    file_prefixes = []
    try:
        for file in os.listdir(directory_path):
            if file.endswith(".npy"):
                file_prefix, _ = os.path.splitext(file)
                if file_prefix != "labels":
                    file_prefixes.append(file_prefix)
        return file_prefixes
    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        return []

def compute_corrupted_embeddings_all(dataroot, level=5, tesize=10000):

    corrupted_data_dict = {}
    file_prefixes = get_file_prefixes(dataroot)
    for corruption in file_prefixes:
        npy_path = os.path.join(dataroot, f'{corruption}.npy')
        teset_raw = np.load(npy_path)
        teset_raw = teset_raw[(level-1)*tesize: level*tesize]
        
        teset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=False, transform=transform)
        teset.data = teset_raw
        corrupted_loader = torch.utils.data.DataLoader(teset, batch_size=128, shuffle=False, num_workers=0)
        embeddings_list, labels_list = [], []
        with torch.no_grad():
            for images, target in corrupted_loader:
                images = images.to(device)
                target = target.to(device)
                outputs = embedder(images)
                embeddings_list.append(outputs.view(outputs.size(0), -1))
                labels_list.append(target)
        corrupted_embeddings = torch.cat(embeddings_list)
        corrupted_labels = torch.cat(labels_list)
        corrupted_data_dict[corruption] = (corrupted_embeddings, corrupted_labels)
        print(f"Computed embeddings for corruption: {corruption}")
    return corrupted_data_dict

def evaluate_corrupted_data_cached(model, corrupted_data_dict):
    resnet50_dict = {}
    corruptList = []
    for corruption, (data_tensor, label_tensor) in corrupted_data_dict.items():
        dataset = torch.utils.data.TensorDataset(data_tensor.to(device), label_tensor.to(device))
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in loader:
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        acc = 100 * correct / total
        print(f'corruption {corruption}, Final Test Accuracy: {acc:.2f}%')
        corruptList.append(acc)
        resnet50_dict[corruption] = acc
    corruption_mean = sum(corruptList) / len(corruptList) if corruptList else 0
    resnet50_dict["corruption_mean"] = corruption_mean
    return resnet50_dict

def add_train_val_test_acc(result_dict, model_name, train_acc, val_acc=None, test_acc=None, full=False):
    result_dict['train_acc'] = train_acc
    if not full:
        result_dict['val_acc'] = val_acc
    result_dict['test_acc'] = test_acc
    result_dict['model'] = model_name
    return result_dict

# ====================================================
# Main 
# ====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process which dataset to run')
    parser.add_argument('-ascending', '--ascending', help='ascending', nargs='?', type=str, default='false')
    parser.add_argument('-num', '--num', help='number of data', nargs='?', type=int, default=30000)
    parser.add_argument('-plus_n', '--plus_n', help='plus_n', nargs='?', type=int, default=0)
    cmd_args = vars(parser.parse_args())
    ascending = True if cmd_args['ascending'] == "true" else False
    num = cmd_args['num']
    plus_n = cmd_args['plus_n']
    
    dataval_df = pd.read_csv("./datavalues/save_dataval.csv")
    df_concat = dataval_df
    wholeindex = dataval_df['indices'].unique()
    valid_indices = list(set(range(4000)) - set(wholeindex))
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    unselected_data = torch.utils.data.Subset(trainset, valid_indices)
    
    
    validation_loader = torch.utils.data.DataLoader(unselected_data, batch_size=128, shuffle=False, num_workers=0)
    embeddings_val, labels_val = compute_embeddings(validation_loader)
    val_dataset   = torch.utils.data.TensorDataset(embeddings_val, labels_val)
    validation_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
        
    dataroot = "../data_files/cifar10/cifar10C"
    corrupted_data_dict = compute_corrupted_embeddings_all(dataroot, level=5, tesize=10000)    
    dictList = []
    for method in dataval_df['Unnamed: 0'].unique():
        train_loader, test_loader = make_trainloader(method, df_concat, num)
        set_seed()
        model, train_acc, val_acc, test_acc = train_model(train_loader,validation_dataloader, test_loader, method, embeddings_val.shape[1])
        set_seed()
        corrupted_dict = evaluate_corrupted_data_cached(model, corrupted_data_dict)
        finaldict = add_train_val_test_acc(corrupted_dict, method, train_acc, val_acc, test_acc)

        dictList.append(finaldict)
    performance_df = pd.DataFrame(dictList)
    
    directory = './removal/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f'performance_ascending_{ascending}_num_{num}.csv')
    performance_df.to_csv(file_path, index=False)
    