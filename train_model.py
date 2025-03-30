import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader

from gcn_image_tagger import ImageGCN, create_adjacency_matrix, ImageTaggingDataset, ImageGCNTrainer

# Import the COCO data loading function from evaluate_coco.py
from evaluate_coco import load_coco_data, NUM_CLASSES

def train_model_on_coco(coco_dir, output_dir='checkpoints', batch_size=16, 
                       num_epochs=30, learning_rate=0.001, device='cuda'):
    """
    Train ImageGCN model on COCO dataset
    
    Args:
        coco_dir: Path to COCO dataset directory
        output_dir: Directory to save model checkpoints
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to use (cuda/cpu)
        
    Returns:
        model: Trained model
        best_val_accuracy: Best validation accuracy achieved
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training and validation data
    print("Loading training data...")
    train_image_paths, train_labels, category_names = load_coco_data(coco_dir, split='train2017')
    
    print("Loading validation data...")
    val_image_paths, val_labels, _ = load_coco_data(coco_dir, split='val2017')
    
    # Create transforms with data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and data loaders
    train_dataset = ImageTaggingDataset(train_image_paths, train_labels, transform=train_transform)
    val_dataset = ImageTaggingDataset(val_image_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create adjacency matrix from training labels
    print("Creating adjacency matrix from label co-occurrences...")
    adj_matrix = create_adjacency_matrix(train_labels, threshold=0.2)
    
    # Initialize model
    print("Initializing model...")
    model = ImageGCN(
        num_classes=NUM_CLASSES,
        feature_dim=2048,  # ResNet-50
        hidden_dim=512,
        dropout=0.5,
        use_vit=False,  # Use ResNet-50
        device=device
    )
    
    # Initialize trainer
    trainer = ImageGCNTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=5e-4
    )
    
    # Train model
    print(f"Training model for {num_epochs} epochs...")
    best_val_accuracy = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        adj_matrix=adj_matrix,
        num_epochs=num_epochs
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Print summary
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model checkpoints saved to {output_dir}")
    
    return model, best_val_accuracy

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ImageGCN on COCO dataset')
    parser.add_argument('--coco-dir', type=str, required=True, 
                        help='Path to COCO dataset directory')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    train_model_on_coco(
        coco_dir=args.coco_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device
    )

if __name__ == "__main__":
    main() 