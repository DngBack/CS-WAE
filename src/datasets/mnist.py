"""
Dataset loading and preprocessing utilities
"""
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ..config import config


def get_mnist_loaders(data_dir='./data', batch_size=None, num_workers=None):
    """
    Create MNIST train and test data loaders
    
    Args:
        data_dir (str): Directory to store/load MNIST data
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    batch_size = batch_size or config.batch_size
    num_workers = num_workers or (2 if os.name == 'nt' else 4)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = datasets.MNIST(
        data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, 
        train=False, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def get_data_info():
    """Get information about the dataset"""
    return {
        'name': 'MNIST',
        'input_shape': (1, 28, 28),
        'num_classes': 10,
        'train_size': 60000,
        'test_size': 10000
    }
