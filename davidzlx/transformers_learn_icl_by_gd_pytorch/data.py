import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        Custom Dataset for loading data.
        
        Args:
            data_path (str): Path to the data file or directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = []
        self.transform = transform
        
        # Example: Assuming data_path is a JSON file with a list of samples
        if os.path.isfile(data_path):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        elif os.path.isdir(data_path):
            # Load all JSON files in the directory
            for filename in os.listdir(data_path):
                if filename.endswith('.json'):
                    with open(os.path.join(data_path, filename), 'r') as f:
                        self.data.extend(json.load(f))
        else:
            raise ValueError(f"Invalid data_path: {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieve a sample by index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            dict: A dictionary containing input and target tensors.
        """
        sample = self.data[idx]
        input_data = sample['input']  # Adjust based on your data structure
        target = sample['target']     # Adjust based on your data structure
        
        # Convert to tensors
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        if self.transform:
            input_tensor = self.transform(input_tensor)
        
        return input_tensor, target_tensor

def get_dataloader(data_path, batch_size=32, shuffle=True, transform=None, num_workers=4):
    """
    Create a DataLoader for the dataset.
    
    Args:
        data_path (str): Path to the data file or directory.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        transform (callable, optional): Optional transform to be applied on a sample.
        num_workers (int): Number of subprocesses to use for data loading.
    
    Returns:
        DataLoader: PyTorch DataLoader.
    """
    dataset = CustomDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader