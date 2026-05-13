import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from model import SceneClassifier

# Set Matplotlib backend to Agg for non-interactive plotting
import matplotlib
matplotlib.use('Agg')

def load_dataset(split, data_dir='scene_classifier/data'):
    data = np.load(f'{data_dir}/{split}_data.npy')
    labels = np.load(f'{data_dir}/{split}_labels.npy')
    # Add channel dimension (B, 1, 64, 64)
    data = torch.from_numpy(data).unsqueeze(1)
    labels = torch.from_numpy(labels)
    return TensorDataset(data, labels)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_set = load_dataset('train')
    val_set = load_dataset('val')
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # Initialize model, loss, optimizer, scheduler
    model = SceneClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 60
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Logging structures
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    checkpoint_dir = 'scene_classifier/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Header for the epoch table
    print(f"{'Epoch':<6} | {'Tr Loss':<8} | {'Tr Acc':<7} | {'Val Loss':<8} | {'Val Acc':<7}")
    print("-" * 50)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        scheduler.step()
        
        # Store history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Print table row
        print(f"{epoch+1:<6} | {epoch_train_loss:<8.4f} | {epoch_train_acc:<7.2%} | {epoch_val_loss:<8.4f} | {epoch_val_acc:<7.2%}")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), f'{checkpoint_dir}/best_model.pth')

    # Plotting
    plot_training_results(history)
    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.2%}")

def plot_training_results(history):
    plots_dir = 'scene_classifier/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', color='#ff7f0e', linestyle='--', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', color='#2ca02c', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Acc', color='#d62728', linestyle='--', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/training_curves.png', dpi=150)
    plt.close()

if __name__ == '__main__':
    train()
