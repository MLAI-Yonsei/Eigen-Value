import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm

# Set device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def skip_none_collate(batch):
    """Collate function that skips None values in batch."""
    batch = [x for x in batch if x is not None]
    if not batch:
        return torch.empty(0), torch.empty(0)
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.tensor(labels).long().view(-1)
    return images, labels

def train_logistic_regression(features, labels, num_epochs=30, lr=0.001):
    """Train logistic regression model."""
    set_seed()
    input_dim = features.size(1)
    num_classes = 345
    logreg_model = nn.Linear(input_dim, num_classes).to(device)
    optimizer = optim.Adam(logreg_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(num_epochs)):
        logreg_model.train()
        optimizer.zero_grad()
        outputs = logreg_model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return logreg_model

def evaluate_model(model, features, labels):
    """Evaluate model accuracy."""
    set_seed()
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        _, preds = torch.max(outputs, dim=1)
        acc = (preds == labels).float().mean().item()
    return acc

def is_valid_image(path):
    """Check if image file is valid."""
    try:
        with Image.open(path).convert("RGB") as img:
            img.verify()
        return True
    except:
        return False

def load_embeddings_and_labels(embedding_dir, domain):
    """Load embeddings and labels for a specific domain."""
    try:
        embeddings = torch.load(os.path.join(embedding_dir, f'{domain}/embeddings.pt'), map_location=device)
        labels = torch.load(os.path.join(embedding_dir, f'{domain}/labels.pt'), map_location=device)
        return embeddings, labels
    except Exception as e:
        print(f"Error loading data for domain {domain}: {str(e)}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Amazon Review Domain Adaptation")
    parser.add_argument('--ascending', type=str, default='false', help='(true - remove high/false - remove low)')
    parser.add_argument('--num', type=int, default=800, help='number of samples to select')
    parser.add_argument('--datavalues', type=str, 
                       help='path to datavalues csv file. Use {domain} as placeholder for domain name')
    parser.add_argument('--save_dir', type=str, 
                       help='directory to save results. Use {domain} as placeholder for domain name')
    parser.add_argument('--embedding_dir', type=str, 
                       help='directory to save results. Use {domain} as placeholder for domain name')
    args = parser.parse_args()
    
    # Initialize parameters
    ascending = True if args.ascending.lower() == "true" else False
    num = args.num
    datavalues_path_template = args.datavalues
    save_dir_template = args.save_dir
    
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    embedding_dir = args.embedding_dir
    
    # Set random seed
    set_seed()
    
    for test_domain in domains:
        train_domains = [d for d in domains if d != test_domain]
        print(f"\n==== LOO: Test domain: {test_domain}, Train domains: {train_domains} ====")
        
        # Load train data
        train_embeddings = []
        train_labels = []
        for d in train_domains:
            emb, lab = load_embeddings_and_labels(embedding_dir, d)
            if emb is not None and lab is not None:
                train_embeddings.append(emb)
                train_labels.append(lab)
        
        if not train_embeddings:
            print(f"Skipping {test_domain} due to missing train data")
            continue
            
        # Concatenate all train embeddings and labels
        train_embeddings = torch.cat(train_embeddings, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
            
        # Load test data
        test_embeddings, test_labels = load_embeddings_and_labels(embedding_dir, test_domain)
        if test_embeddings is None or test_labels is None:
            print(f"Skipping {test_domain} due to missing test data")
            continue
            
        # Load data values
        try:
            dataval_df = pd.read_csv(datavalues_path_template.format(domain=test_domain))
            methods = dataval_df['Unnamed: 0'].unique().tolist()
        except Exception as e:
            print(f"Error loading data values for {test_domain}: {str(e)}")
            continue
            
        performance_results = []
        
        for method in methods:
            print(f"\n--- Method: {method} ---")
            
            method_df = dataval_df[dataval_df['Unnamed: 0'] == method]
            sorted_df = method_df.sort_values(by='data_values', ascending=ascending)
            selected_df = sorted_df.head(num)
            selected_indices = selected_df['indices'].tolist()
            
            # Train and evaluate
            try:
                embeddings_selected = train_embeddings[selected_indices]
                labels_selected = train_labels[selected_indices]
                
                logreg_model = train_logistic_regression(embeddings_selected, labels_selected,
                                                       num_epochs=1000, lr=0.001)
                
                train_acc = evaluate_model(logreg_model, train_embeddings, train_labels)
                test_acc = evaluate_model(logreg_model, test_embeddings, test_labels)
                
                performance_results.append({
                    "test_domain": test_domain,
                    "method": method,
                    "num": num,
                    "train_acc": train_acc,
                    "test_acc": test_acc
                })
            except Exception as e:
                print(f"Error processing method {method}: {str(e)}")
                continue
        
        # Save results
        if performance_results:
            performance_df = pd.DataFrame(performance_results)
            output_dir = save_dir_template.format(domain=test_domain)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'performance_ascending_{ascending}_num_{num}.csv')
            performance_df.to_csv(output_path, index=False)
            print(f"\nPerformance results saved to {output_path}")

if __name__ == "__main__":
    main()