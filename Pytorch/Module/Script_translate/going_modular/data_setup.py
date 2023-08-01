
import os

import torch
import torch.utils.data.DataLoader as DataLoader
import torch.nn as nn
import torchvision.transform as transform
import torchvision.dataset.ImageFolder as ImageFolder
from typing import Tuple, List

NUM_WORKERS = os.cpu_count()

def create_dataloader(train_dir: str,
                      test_dir: str,
                      train_transform: transform.Compose | nn.Module = None,
                      test_transform: transform.Compose | nn.Module = None,
                      batch_size: int, 
                      num_workers: int = NUM_WORKERS) -> Tuple[DataLoader, DataLoader, List[str]]:
    """ Create train and test DataLoader
    Pass train_dir and test_dir directories to get train and validation DataLoader 
    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
    Example usage:
          train_dataloader, test_dataloader, class_names = \
            = create_dataloaders(train_dir=path/to/train_dir,
                                 test_dir=path/to/test_dir,
                                 transform=some_transform,
                                 batch_size=32,
                                 num_workers=4)
    """
    
    
    # Create dataset using ImageFolder
    
    # Train Dataset from train_dir directory
    train_dataset = ImageFolder(root=train_dir,
                                transform=train_transform,
                                target_transform=None)

    # Test Dataset from test_dir directory
    test_dataset = ImageFolder(root=test_dir,
                               transform=test_transform,
                               target_transform=None)
    
    # Get class names of target
    class_names = train_dataset.classes
    
    # Create train, test DataLoader
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    
    test_dataloader = DataLoader(dataset=test_dataset, 
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)
    
    return train_dataloader, test_dataloader, class_names
