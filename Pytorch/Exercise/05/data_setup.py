import argparse
import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

num_workers = os.cpu_counts()

def create_dataset(train_dir: str, test_dir: str, train_transform: Compose, test_transform: Compose):
    
    train_dataset = ImageFolder(root=train_dir, transform=train_transform, target_transform=None)
    test_dataset = ImageFolder(root=test_dir, transform=test_transform, target_transform=None)
    return train_dataset, test_dataset

def create_dataloader(train_dataset: Dataset, test_dataset: Dataset, batch_size: int = 8, num_workers: int = num_workers):
    class_names = test_dataset.classes
    
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers=num_workers)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers=num_workers)
    
    return train_dataloader, test_dataloader, class_names
