import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import re
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
def skip_none_collate(batch):
    
    batch = [x for x in batch if x is not None]
    if not batch:
        
        return torch.empty(0), torch.empty(0)
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.tensor(labels).long().view(-1)
    
    return images, labels


def train_logistic_regression(features, labels, num_epochs=30, lr=0.001):
    set_seed()
    input_dim = features.size(1)
    # num_classes = len(torch.unique(labels))
    num_classes = 1000
    logreg_model = nn.Linear(input_dim, num_classes).to(device)
    optimizer = optim.Adam(logreg_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(num_epochs)):
        logreg_model.train()
        optimizer.zero_grad()
        outputs = logreg_model(features)
        loss = criterion(outputs, labels)
        l2_lambda = 0.01
        l2_norm = sum(param.pow(2.0).sum() for param in logreg_model.parameters())
        loss = loss + l2_lambda * l2_norm
        loss.backward()
        optimizer.step()
    return logreg_model

def evaluate_model(model, features, labels):
    set_seed()
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        _, preds = torch.max(outputs, dim=1)
        acc = (preds == labels).float().mean().item()
    return acc
def _is_valid_image( path_):
    try:
        with Image.open(path_).convert("RGB") as img:
            img.verify()
        return True
    except:
        return False


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    parser = argparse.ArgumentParser(description="VLCS")
    parser.add_argument('--ascending', type=str, default='false', help='(true - remove high/false - remove low)')
    parser.add_argument('--num', type=int, default=800, help='data number')
    parser.add_argument('--datavalues', type=str, 
                       help='path to datavalues csv file. Use {domain} as placeholder for domain name')
    parser.add_argument('--save_dir', type=str,
                       help='directory to save results. Use {domain} as placeholder for domain name')
    parser.add_argument('--embedding_dir', type=str, 
                       help='directory to save results. Use {domain} as placeholder for domain name')
    args = parser.parse_args()
    
    ascending = True if args.ascending.lower() == "true" else False
    num = args.num  
    datavalues_path_template = args.datavalues
    save_dir_template = args.save_dir


    embeddings = torch.load(args.embedding_dir + '/embeddings.pt', map_location=device)
    labels = torch.load(args.embedding_dir + '/labels.pt', map_location=device)
    
    

    embeddings_IN_V2 = torch.load(args.embedding_dir + '/wds_imagenetv2/embeddings.pt', map_location=device)
    labels_IN_V2 = torch.load(args.embedding_dir + '/wds_imagenetv2/labels.pt', map_location=device)
    

    embeddings_S = torch.load(args.embedding_dir + '/ImageNet-S/embeddings.pt', map_location=device)
    labels_S = torch.load(args.embedding_dir + '/ImageNet-S/labels.pt', map_location=device)

    embeddings_R = torch.load(args.embedding_dir + '/ImageNet-R/embeddings.pt', map_location=device)
    labels_R = torch.load(args.embedding_dir + '/ImageNet-R/labels.pt', map_location=device)
    

    embeddings_A = torch.load(args.embedding_dir + '/ImageNet-A/embeddings.pt', map_location=device)
    labels_A = torch.load(args.embedding_dir + '/ImageNet-A/labels.pt', map_location=device)
    

    dataval_df = pd.read_csv(datavalues_path_template)
    methods = dataval_df['Unnamed: 0'].unique().tolist()
    
    performance_results=[]

    for method in methods:
        print(f"\n--- Method: {method} ---")
    
        method_df = dataval_df[dataval_df['Unnamed: 0'] == method]
        
        
    
    

        sorted_df = method_df.sort_values(by='data_values', ascending=ascending)
        selected_df = sorted_df.head(num)
        selected_indices = selected_df['indices'].tolist()

        
    
        
        embeddings_selected = embeddings[selected_indices]
        labels_selected = labels[selected_indices]

        logreg_model = train_logistic_regression(embeddings_selected, labels_selected,
                                                     num_epochs=1000, lr=0.001)
    
        train_acc = evaluate_model(logreg_model, embeddings, labels)
        v2_acc = evaluate_model(logreg_model, embeddings_IN_V2, labels_IN_V2)
        sketch_acc = evaluate_model(logreg_model, embeddings_S, labels_S)
        real_acc = evaluate_model(logreg_model, embeddings_R, labels_R)
        art_acc = evaluate_model(logreg_model, embeddings_A, labels_A)
            
        performance_results.append({
            "train_acc": train_acc,
            "method": method,
            "num": num,
            "v2_acc":v2_acc,
            "sketch_acc": sketch_acc,
            "real_acc": real_acc,
            "art_acc": art_acc,
            "mean_acc": (v2_acc + sketch_acc + real_acc + art_acc) / 4
        })

    performance_df = pd.DataFrame(performance_results)
    output_dir = save_dir_template
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'performance_ascending_{ascending}_num_{num}.csv')
    performance_df.to_csv(output_path, index=False)
    print(f"\nPerformance results saved to {output_path}")
