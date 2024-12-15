import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from transformer import Transformer, TransformerConfig
from data import get_dataloader
import os

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The Transformer model.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run the training on.
    
    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs, _, _ = model(inputs, is_training=True)
        
        loss = criterion(outputs.squeeze(), targets.squeeze())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss

def validate_epoch(model, dataloader, criterion, device):
    """
    Validate the model for one epoch.
    
    Args:
        model (nn.Module): The Transformer model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the validation on.
    
    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs, _, _ = model(inputs, is_training=False)
            
            loss = criterion(outputs.squeeze(), targets.squeeze())
            total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training checkpoint.
    
    Args:
        state (dict): State dictionary containing model and optimizer states.
        filename (str): Path to save the checkpoint.
    """
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    """
    Load the training checkpoint.
    
    Args:
        model (nn.Module): The Transformer model.
        optimizer (Optimizer): Optimizer.
        filename (str): Path to the checkpoint.
    
    Returns:
        int: Starting epoch.
        float: Best validation loss.
    """
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return start_epoch, best_val_loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, float('inf')

def main():
    # Configuration
    config = TransformerConfig(
        num_heads=4,
        widening_factor=4,
        num_layers=6,
        key_size=64,
        embedding_size=256,
        output_size=1,
        in_context_length=50,
        dropout_rate=0.1,
        vocab_size=10000,  # Set according to your data
        vocab_token_dim=256,
        return_logits=False
    )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    model = Transformer(config).to(device)
    
    # DataLoader
    train_loader = get_dataloader(
        data_path='path/to/train_data.json',
        batch_size=64,
        shuffle=True,
        transform=None,  # Add transforms if needed
        num_workers=8
    )
    
    val_loader = get_dataloader(
        data_path='path/to/val_data.json',
        batch_size=64,
        shuffle=False,
        transform=None,  # Add transforms if needed
        num_workers=8
    )
    
    # Loss and Optimizer
    criterion = nn.MSELoss()  # Change based on your task (e.g., CrossEntropyLoss for classification)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Optionally load from a checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = 'checkpoint.pth.tar'
    if os.path.exists(checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path)
    
    # Training Loop
    num_epochs = 50
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, filename=checkpoint_path if is_best else f'checkpoint_epoch_{epoch+1}.pth.tar')
        
        if is_best:
            print("New best model found and saved.")
        
        # Early stopping or other callbacks can be added here

if __name__ == "__main__":
    main()