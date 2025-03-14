import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset,Subset


from PIL import Image
from opendataval.experiment.exper_methods import (
    discover_corrupted_sample,
    noisy_detection,
    remove_high_low,
    save_dataval
)
import matplotlib.pyplot as plt

from opendataval.dataloader import Register, DataFetcher, mix_labels
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
    weightsingularDataOob
)
from opendataval.experiment import ExperimentMediator
from opendataval.model.logistic_regression import LogisticRegression
from opendataval.experiment.exper_methods import save_dataval

from torchvision.models.resnet import resnet50, ResNet50_Weights
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


class VLCSDataset(Dataset):
    def __init__(self, label_files, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []

        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    img_path, label = line.strip().split()
                    full_img_path = os.path.join(self.data_root, img_path)
                    self.samples.append((full_img_path, int(label)))
        
        
        valid_samples = []
        for (path_, label_) in self.samples:
            if self._is_valid_image(path_):
                valid_samples.append((path_, label_))
            else:
                print(f"[Dataset Init] Skipped invalid image: {path_}")
        self.samples = valid_samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(idx)

        
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


real_transform_vlcs = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


ALL_TRAIN_FILES_VLCS = [
    '../data/VLCS/vlcs_label/VOC2007/VOC2007_train.txt',
    '../data/VLCS/vlcs_label/LabelMe/LabelMe_train.txt',
    '../data/VLCS/vlcs_label/Caltech101/Caltech101_train.txt',
    '../data/VLCS/vlcs_label/SUN09/SUN09_train.txt'
]


def get_excluded_and_included_domains_vlcs(exclude_idx):
    excluded_file = ALL_TRAIN_FILES_VLCS[exclude_idx]
    included_files = [f for i, f in enumerate(ALL_TRAIN_FILES_VLCS) if i != exclude_idx]
    
    
    import re
    pattern = r'/([^/]+)_train\.txt'  
    match = re.search(pattern, excluded_file)
    if match:
        excluded_domain = match.group(1)
    else:
        excluded_domain = f"domain_{exclude_idx}"
    return included_files, excluded_domain
def skip_none_collate(batch):
    
    batch = [x for x in batch if x is not None]
    if not batch:
    
        return torch.empty(0), torch.empty(0)
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.tensor(labels).long().view(-1)
    

    return images, labels

def main_vlcs():
    
    data_root = '../data/VLCS/vlcs_data'
    noise_rate = 0.1  
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedder = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    embedder.fc = nn.Identity()
    embedder = embedder.to(device)

    
    for exclude_idx in range(4):
        
        set_seed()
        train_label_files, excluded_domain = get_excluded_and_included_domains_vlcs(exclude_idx)
        
        
      
        dataset = VLCSDataset(
            label_files=train_label_files,
            data_root=data_root,
            transform=real_transform_vlcs
        )
        damaged_indices = [504, 557, 558,1592, 1593, 1758,3933,4968,5133]
        
        total = len(dataset)
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

        
        subset_dataset = Subset(dataset, selected_indices)

        
        loader = DataLoader(subset_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=skip_none_collate)

        
    
        embedder = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device) # In case of LabelMe
        embedder.heads = nn.Identity()  
        embedder.eval()
        
        
        embeddings = []
        labels = []
        with torch.no_grad():
            for images, targets in loader:
                

                images = images.to(device)
                outputs = embedder(images)
                embeddings.append(outputs.cpu())
                labels.append(targets)
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        
        
        
        dataset_name = f"VLCS_excluded_{excluded_domain}"
        pd_dataset = Register(dataset_name=dataset_name, one_hot=True).from_data(embeddings.numpy(), labels.numpy())
        
        
        total = embeddings.shape[0]
        indices = np.arange(total)
        
        train_size = 2000
        valid_size = 1000


        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        test_indices = indices[train_size + valid_size:]
        
        
        fetcher = (
            DataFetcher(dataset_name, '../data_files/', False)
            .split_dataset_by_indices(train_indices=train_indices,
                                      valid_indices=valid_indices,
                                      test_indices=test_indices)
            .noisify(mix_labels, noise_rate=noise_rate)
        )
        
        
        pred_model = LogisticRegression(fetcher.covar_dim[0], num_classes=5)
        train_kwargs = {"epochs": 3, "batch_size": 100, "lr": 0.01}
        exper_med = ExperimentMediator(fetcher, pred_model, train_kwargs=train_kwargs, metric_name="accuracy")
        
        
        data_evaluators = [
            RandomEvaluator(),
            SingularLavaEvaluator(),
            singularKNNShapley(k_neighbors=valid_size),
            InfluenceSubsample(num_models=1000),
            KNNShapley(k_neighbors=valid_size),
            DataOob(num_models=800),
            LavaEvaluator(),
            singularDataOob(num_models=800),
            weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = 1),
            weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = -1),
            weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = 0.1),
            weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = -0.1),
            weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = 0.01),
            weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = -0.01),
            weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = 0.001),
            weightsingularDataOob(num_models=800, weight = 1, epsilon_weight = -0.001),
            weightsingularDataOob(num_models=800, weight = 0, epsilon_weight = 1),
            weightsingularDataOob(num_models=800, weight = 0, epsilon_weight = -1),
            weightsingularDataOob(num_models=800, weight = 0, epsilon_weight = 0.1),
            weightsingularDataOob(num_models=800, weight = 0, epsilon_weight = -0.1),
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
            ModelDeviationEvaluator()
        ]

        exper_med = exper_med.compute_data_values(data_evaluators=data_evaluators)
        

        output_dir = f"./datavalues/{excluded_domain}"
        exper_med.set_output_directory(output_dir)
        exper_med.evaluate(save_dataval, save_output=True)
        
if __name__ == "__main__":
    overall_start = time.time()
    main_vlcs()
    overall_end = time.time()
    