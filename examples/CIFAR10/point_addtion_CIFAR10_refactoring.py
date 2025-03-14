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

def make_trainloader(model, dataval_df, plus_n=0):
    set_seed()
    
    random_1000_idx = random.sample(list(dataval_df['indices'].unique()), 1000)
    remain_idx = list(set(dataval_df['indices'].unique()) - set(random_1000_idx))
    
    
    remain_df = dataval_df[dataval_df['indices'].isin(remain_idx)]
    grouped_df = remain_df.sort_values(by=['Unnamed: 0', 'data_values'], ascending=False)\
                          .groupby('Unnamed: 0').head(plus_n)[['Unnamed: 0','indices']]
    
    filtered_df = dataval_df[dataval_df['indices'].isin(random_1000_idx)]
    sorted_df = pd.concat([grouped_df, filtered_df], ignore_index=True).copy()
    
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    num_samples = 4000
    set_seed()
    random_train_indices = random.sample(range(len(trainset)), num_samples)
    
    
    num = return_indices(model=f'{model}', df=sorted_df)
    
    wholeindex = dataval_df['indices'].unique()
    valid_indices = list(set(range(num_samples)) - set(wholeindex))
    
    random_trainset = torch.utils.data.Subset(trainset, random_train_indices)
    selected_data = torch.utils.data.Subset(random_trainset, num)
    unselected_data = torch.utils.data.Subset(trainset, valid_indices)
    
    train_loader = torch.utils.data.DataLoader(selected_data, batch_size=128, shuffle=True, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(unselected_data, batch_size=128, shuffle=False, num_workers=0)
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
    embeddings_val, labels_val = compute_embeddings(validation_loader)
    embeddings_test, labels_test = compute_embeddings(test_loader)
    
    
    train_dataset = torch.utils.data.TensorDataset(embeddings_train, labels_train)
    val_dataset = torch.utils.data.TensorDataset(embeddings_val, labels_val)
    test_dataset = torch.utils.data.TensorDataset(embeddings_test, labels_test)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    validation_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return train_dataloader, validation_dataloader, test_dataloader


def prepare_full_dataloaders():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    full_train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    full_test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
    
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
    test_dataset = torch.utils.data.TensorDataset(embeddings_test, labels_test)
    
    full_train_loader_emb = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
    full_test_loader_emb = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
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
        files = os.listdir(directory_path)
        for file in files:
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

    
    lava_train_loader, lava_valdiation_loader, lava_test_loader = make_trainloader("LavaEvaluator()", df_concat, plus_n)
    Influence_train_loader, Influence_valdiation_loader, Influence_test_loader = make_trainloader('InfluenceSubsample(num_models=1000)', df_concat, plus_n)
    oob_train_loader, oob_valdiation_loader, oob_test_loader = make_trainloader('DataOob(num_models=800)', df_concat, plus_n)
    knn_train_loader, knn_valdiation_loader, knn_test_loader = make_trainloader("KNNShapley(k_neighbors=1000)", df_concat, plus_n)
    drge_train_loader, drge_valdiation_loader, drge_test_loader = make_trainloader('ModelDeviationEvaluator()', df_concat, plus_n)
    singInf_train_loader, singInf_valdiation_loader, singInf_test_loader = make_trainloader("SingularLavaEvaluator()", df_concat, plus_n)
    singknn_train_loader, singknn_valdiation_loader, singknn_test_loader = make_trainloader("singularKNNShapley(k_neighbors=1000)", df_concat, plus_n)
    singoob_train_loader, singoob_valdiation_loader, singoob_test_loader = make_trainloader("singularDataOob(num_models=800)", df_concat, plus_n)
    random_train_loader, random_valdiation_loader, random_test_loader = make_trainloader('RandomEvaluator()', df_concat, plus_n)
    
    
    full_train_loader, full_test_loader, input_dim = prepare_full_dataloaders()
    
    
    set_seed()
    lava_model, lava_train_acc, lava_val_acc, lava_test_acc = train_model(lava_train_loader, lava_valdiation_loader, lava_test_loader, "LAVA", input_dim)
    set_seed()
    Influence_model, Influence_train_acc, Influence_val_acc, Influence_test_acc = train_model(Influence_train_loader, Influence_valdiation_loader, Influence_test_loader, "Influence", input_dim)
    set_seed()
    oob_model, oob_train_acc, oob_val_acc, oob_test_acc = train_model(oob_train_loader, oob_valdiation_loader, oob_test_loader, "OOB", input_dim)
    set_seed()
    knn_model, knn_train_acc, knn_val_acc, knn_test_acc = train_model(knn_train_loader, knn_valdiation_loader, knn_test_loader, "KNN", input_dim)
    set_seed()
    drge_model, drge_train_acc, drge_val_acc, drge_test_acc = train_model(drge_train_loader, drge_valdiation_loader, drge_test_loader, "DRGE", input_dim)
    set_seed()
    singInf_model, singInf_train_acc, singInf_val_acc, singInf_test_acc = train_model(singInf_train_loader, singInf_valdiation_loader, singInf_test_loader, "SingLava", input_dim)
    set_seed()
    singknn_model, singknn_train_acc, singknn_val_acc, singknn_test_acc = train_model(singknn_train_loader, singknn_valdiation_loader, singknn_test_loader, "SingKNN", input_dim)
    set_seed()
    singoob_model, singoob_train_acc, singoob_val_acc, singoob_test_acc = train_model(singoob_train_loader, singoob_valdiation_loader, singoob_test_loader, "SingOOB", input_dim)
    set_seed()
    random_model, random_train_acc, random_val_acc, random_test_acc = train_model(random_train_loader, random_valdiation_loader, random_test_loader, "Random", input_dim)
    set_seed()
    full_model, full_train_acc, full_test_acc = train_full_model(full_train_loader, full_test_loader, "Full", input_dim)
    
    
    dataroot = "../data_files/cifar10/cifar10C"
    corrupted_data_dict = compute_corrupted_embeddings_all(dataroot, level=5, tesize=10000)
    
    
    set_seed()
    lava_corrupted_dict = evaluate_corrupted_data_cached(lava_model, corrupted_data_dict)
    lava_finaldict = add_train_val_test_acc(lava_corrupted_dict, "lava", lava_train_acc, lava_val_acc, lava_test_acc)
    
    set_seed()
    Influence_corrupted_dict = evaluate_corrupted_data_cached(Influence_model, corrupted_data_dict)
    Influence_finaldict = add_train_val_test_acc(Influence_corrupted_dict, "Influence", Influence_train_acc, Influence_val_acc, Influence_test_acc)
    
    set_seed()
    oob_corrupted_dict = evaluate_corrupted_data_cached(oob_model, corrupted_data_dict)
    oob_finaldict = add_train_val_test_acc(oob_corrupted_dict, "oob", oob_train_acc, oob_val_acc, oob_test_acc)
    
    set_seed()
    knn_corrupted_dict = evaluate_corrupted_data_cached(knn_model, corrupted_data_dict)
    knn_finaldict = add_train_val_test_acc(knn_corrupted_dict, "knn", knn_train_acc, knn_val_acc, knn_test_acc)
    
    set_seed()
    drge_corrupted_dict = evaluate_corrupted_data_cached(drge_model, corrupted_data_dict)
    drge_finaldict = add_train_val_test_acc(drge_corrupted_dict, "drge", drge_train_acc, drge_val_acc, drge_test_acc)
    
    set_seed()
    singInf_corrupted_dict = evaluate_corrupted_data_cached(singInf_model, corrupted_data_dict)
    singInf_finaldict = add_train_val_test_acc(singInf_corrupted_dict, "singLava", singInf_train_acc, singInf_val_acc, singInf_test_acc)
    
    set_seed()
    singknn_corrupted_dict = evaluate_corrupted_data_cached(singknn_model, corrupted_data_dict)
    singknn_finaldict = add_train_val_test_acc(singknn_corrupted_dict, "singKNN", singknn_train_acc, singknn_val_acc, singknn_test_acc)
    
    set_seed()
    singoob_corrupted_dict = evaluate_corrupted_data_cached(singoob_model, corrupted_data_dict)
    singoob_finaldict = add_train_val_test_acc(singoob_corrupted_dict, "singOOB", singoob_train_acc, singoob_val_acc, singoob_test_acc)
    
    set_seed()
    random_corrupted_dict = evaluate_corrupted_data_cached(random_model, corrupted_data_dict)
    random_finaldict = add_train_val_test_acc(random_corrupted_dict, "random", random_train_acc, random_val_acc, random_test_acc)
    
    set_seed()
    full_corrupted_dict = evaluate_corrupted_data_cached(full_model, corrupted_data_dict)
    full_finaldict = add_train_val_test_acc(full_corrupted_dict, "full", full_train_acc, test_acc=full_test_acc, full=True)
    
    dictList = [lava_finaldict, Influence_finaldict, oob_finaldict, knn_finaldict,
                drge_finaldict, singInf_finaldict, singknn_finaldict,
                singoob_finaldict, random_finaldict, full_finaldict]
    performance_df = pd.DataFrame(dictList)
    
    directory = './point_addition/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f'performance_ascending_{ascending}_plus_n{plus_n}.csv')
    performance_df.to_csv(file_path, index=False)
