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

class VLCSDataset(Dataset):
    def __init__(self, label_files, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        self.damaged_indices = []  
        
        tmp_list = []  
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    img_path, label = line.strip().split()
                    full_img_path = os.path.join(self.data_root, img_path)
                    tmp_list.append((full_img_path, int(label)))
        
        valid_samples = []
        for idx, (path_, label_) in enumerate(tmp_list):
            if self._is_valid_image(path_):
                valid_samples.append((path_, label_))
            else:
                
                self.damaged_indices.append(idx)
                print(f"[Dataset Init] Skipped invalid image idx={idx}: {path_}")
        
        self.samples = valid_samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except OSError:
            print(f"[Warning] Skipping truncated image: {img_path}, idx= {idx}")
            return None  

        if self.transform:
            image = self.transform(image)
        return image, label
    def _is_valid_image(self, path_):
        try:
            with Image.open(path_) as img:
                img.verify()
            return True
        except:
            return False

real_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
augmented_transform = transforms.Compose([
    transforms.Resize((int(224 * 1.25), int(224 * 1.25))),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


ALL_TRAIN_FILES = [
    '../data/VLCS/vlcs_label/VOC2007/VOC2007_train.txt',
    '../data/VLCS/vlcs_label/LabelMe/LabelMe_train.txt',
    '../data/VLCS/vlcs_label/Caltech101/Caltech101_train.txt',
    '../data/VLCS/vlcs_label/SUN09/SUN09_train.txt'
]

ALL_TEST_FILES =  [
    '../data/VLCS/vlcs_label/VOC2007/VOC2007_test.txt',
    '../data/VLCS/vlcs_label/LabelMe/LabelMe_test.txt',
    '../data/VLCS/vlcs_label/Caltech101/Caltech101_test.txt',
    '../data/VLCS/vlcs_label/SUN09/SUN09_test.txt'
]

def get_vlcs_label_files(exclude_idx: int):

    train_files = [f for i, f in enumerate(ALL_TRAIN_FILES) if i != exclude_idx]
    
    test_files = [ALL_TEST_FILES[exclude_idx]]
    pattern = r'/([^/]+)_train\.txt'
    match = re.search(pattern, ALL_TRAIN_FILES[exclude_idx])
    if match:
        excluded_domain = match.group(1)
    else:
        excluded_domain = f"domain_{exclude_idx}"
    return train_files, test_files, excluded_domain


def extract_embeddings(dataloader, model, device):
    model.eval()
    embeddings_list = []
    labels_list = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            emb = model(inputs)
        embeddings_list.append(emb.cpu())
        labels_list.append(labels)
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return embeddings, labels

# ---------------------------

def train_logistic_regression(features, labels, num_epochs=30, lr=0.001):
    set_seed()
    input_dim = features.size(1)
    # num_classes = len(torch.unique(labels))
    num_classes = 5
    logreg_model = nn.Linear(input_dim, num_classes)
    optimizer = optim.Adam(logreg_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        logreg_model.train()
        optimizer.zero_grad()
        outputs = logreg_model(features)
        loss = criterion(outputs, labels)
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
    set_seed()
    parser = argparse.ArgumentParser(description="VLCS")
    parser.add_argument('--ascending', type=str, default='false', help='(true - remove high/false - remove low)')
    parser.add_argument('--num', type=int, default=800, help='data number')
    

    
    args = parser.parse_args()
    
    ascending = True if args.ascending.lower() == "true" else False
    num = args.num  
    # root_dir = args.root_dir
    
    
    
    
    for exclude_idx in range(4):
    
        performance_results = []
        train_files, test_files, excluded_domain = get_vlcs_label_files(exclude_idx)
    

        damaged_indices = [504, 558, 1592, 1593, 1758,3933,4968,5133] # index of truncated images 

        
        print(f"\n=== Excluded domain: {excluded_domain} ===")
    
        dataval_csv_path = f'../datavalues/ViT/{excluded_domain}/save_dataval.csv' 
        dataval_df = pd.read_csv(dataval_csv_path)
        
        
        unique_indices = dataval_df['indices'].unique()
        if len(unique_indices) > 1000:
            set_seed()
            sampled_idx = random.sample(list(unique_indices), 1000)
        else:
            sampled_idx = list(unique_indices)
        
        # sampled_idx = sorted(sampled_idx)
        sampled_idx = unique_indices
        
        raw_train = VLCSDataset(
            label_files=train_files,
            data_root='../data/VLCS/vlcs_data',
            transform=real_transform
        )

        total = len(raw_train)
        indices = np.arange(total)
        set_seed()  
        np.random.shuffle(indices)

       
        damaged_indices = set(damaged_indices)  

        
        selected_indices = [idx for idx in indices if idx not in damaged_indices]

        
        if len(selected_indices) < 4000:
            remaining_indices = set(indices) - set(selected_indices) - damaged_indices
            additional_needed = 4000 - len(selected_indices)

            if len(remaining_indices) >= additional_needed:
                
                new_samples = np.random.choice(list(remaining_indices), additional_needed, replace=False).tolist()
                selected_indices.extend(new_samples)
            else:
                
                selected_indices.extend(list(remaining_indices))


        
        selected_indices = selected_indices[:4000]

        train_dataset = Subset(raw_train, selected_indices)

        
        
        fixed_train_subset = Subset(train_dataset, sampled_idx)
        
        
        fixed_train_loader = DataLoader(fixed_train_subset, batch_size=16, shuffle=False, num_workers=0, collate_fn=skip_none_collate)
        
        val_remain_indices = list(set(list(range(len(train_dataset)))) - set(sampled_idx))

        val_subset = Subset(train_dataset, val_remain_indices)

        validation_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=0, collate_fn=skip_none_collate)
        
        
        test_dataset = VLCSDataset(label_files=test_files,
                                   data_root='../data/VLCS/vlcs_data',
                                   transform=real_transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0,  collate_fn=skip_none_collate)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        embedding_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device)
        embedding_model.heads = nn.Identity() 
        embedding_model.eval()
        
        
        
        
        train_embeddings, train_labels = extract_embeddings(fixed_train_loader, embedding_model, device)
        val_embeddings, val_labels = extract_embeddings(validation_loader, embedding_model, device)
        test_embeddings, test_labels = extract_embeddings(test_loader, embedding_model, device)
        
        
        index_to_position = {idx: pos for pos, idx in enumerate(sampled_idx)}
        
        
        methods = dataval_df['Unnamed: 0'].unique().tolist()
        
        for method in methods:
            print(f"\n--- Method: {method} ---")
        
            method_df = dataval_df[dataval_df['Unnamed: 0'] == method]
            method_df = method_df[method_df['indices'].isin(sampled_idx)]
            
        
            if num == 1000:
                selected_indices = sampled_idx
            else:
        
                sorted_df = method_df.sort_values(by='data_values', ascending=ascending)
                selected_df = sorted_df.head(num)
                selected_indices = selected_df['indices'].tolist()
        
                selected_indices = sorted(selected_indices, key=lambda x: index_to_position[x])
            
        
            selected_positions = [index_to_position[idx] for idx in selected_indices]
            
        
            train_emb_selected = train_embeddings[selected_positions]
            train_lab_selected = train_labels[selected_positions]
            
            print(f"Training logistic regression with method {method} on {len(selected_positions)} samples")
            
        
            logreg_model = train_logistic_regression(train_emb_selected, train_lab_selected,
                                                     num_epochs=100, lr=0.001)
        
            val_acc = evaluate_model(logreg_model, val_embeddings, val_labels)
            test_acc = evaluate_model(logreg_model, test_embeddings, test_labels)
            
            performance_results.append({
                "exclude_domain": excluded_domain,
                "method": method,
                "num": num,
                "val_acc":val_acc,
                "test_acc": test_acc
            })

        performance_df = pd.DataFrame(performance_results)
        output_dir = f'./removal/{excluded_domain}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f'performance_ascending_{ascending}_num_{num}.csv')
        performance_df.to_csv(output_path, index=False)
        print(f"\nPerformance results saved to {output_path}")
        