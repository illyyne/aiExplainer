"""
Alien Species Classification Model Training
PyTorch-based CNN for classifying 4 alien species
Fixed for macOS compatibility
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Alien Dataset Class
class AlienDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# CNN Model
class AlienCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(AlienCNN, self).__init__()
        
        # Use pretrained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Modify the final layer for our classes
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # For heatmap generation (we'll use features before final FC)
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        # Get features for heatmap
        features = self.features(x)
        
        # Classification
        pooled = self.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        output = self.classifier(pooled)
        
        return output, features

def load_dataset(images_dir='images/aliens', csv_path='images/class/classification.csv'):
    """Load alien dataset from images and CSV"""
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Class mapping
    classes = ['Krythik', 'Abyssal', 'Anthroide', 'Fluffony']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Find images
    images_dir = Path(images_dir)
    image_paths = []
    labels = []
    
    for _, row in df.iterrows():
        img_num = str(row['Image']).zfill(3)
        label = row['Label']
        
        # Try different extensions
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            img_path = images_dir / f"{img_num}{ext}"
            if img_path.exists():
                image_paths.append(str(img_path))
                labels.append(class_to_idx[label])
                break
    
    print(f"Loaded {len(image_paths)} images")
    print(f"Classes: {classes}")
    
    return image_paths, labels, classes, class_to_idx

def train_model(model, train_loader, val_loader, num_epochs=30, lr=0.001):
    """Train the model"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/alien_classifier_best.pth')
            print(f'✓ Saved best model (val_acc: {val_acc:.2f}%)')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Best Val Acc: {best_val_acc:.2f}%')
        print('-' * 60)
    
    return history

def plot_training_history(history):
    """Plot training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
    print('Saved training history plot to models/training_history.png')

def main():
    """Main training pipeline"""
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    image_paths, labels, classes, class_to_idx = load_dataset()
    
    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Train samples: {len(train_paths)}")
    print(f"Val samples: {len(val_paths)}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = AlienDataset(train_paths, train_labels, train_transform)
    val_dataset = AlienDataset(val_paths, val_labels, val_transform)
    
    # Create dataloaders - FIX: Set num_workers=0 for macOS compatibility
    print("\nCreating dataloaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,  # Reduced batch size for CPU
        shuffle=True, 
        num_workers=0,  # FIXED: 0 workers to avoid multiprocessing issues on macOS
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8,  # Reduced batch size
        shuffle=False, 
        num_workers=0,  # FIXED: 0 workers
        pin_memory=False
    )
    
    # Create model
    print("\nCreating model...")
    model = AlienCNN(num_classes=len(classes)).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nStarting training...")
    print("Note: Training on CPU may take 15-30 minutes")
    print("Consider using GPU for faster training\n")
    
    history = train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001)
    
    # Save final model
    torch.save(model.state_dict(), 'models/alien_classifier_final.pth')
    print("\nSaved final model to models/alien_classifier_final.pth")
    
    # Save metadata
    metadata = {
        'classes': classes,
        'class_to_idx': class_to_idx,
        'num_classes': len(classes),
        'input_size': [224, 224],
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Saved metadata to models/metadata.json")
    
    # Save training history
    with open('models/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot history
    plot_training_history(history)
    
    print("\n✓ Training complete!")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")

if __name__ == '__main__':
    main()