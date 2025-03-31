import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from voc_dataset import get_voc_dataset, load_voc_data, NUM_CLASSES
from gcn_image_tagger_simplified import ImageGCNSimple, create_adjacency_matrix, ImageGCNTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train ImageGCN model on VOC 2007 dataset')
    parser.add_argument('--voc-dir', type=str, required=True,
                        help='Path to VOC2007 dataset directory')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for training')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Hidden dimension for GCN layers')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--use-vit', action='store_true',
                        help='Use Vision Transformer instead of ResNet')
    parser.add_argument('--adj-threshold', type=float, default=0.2,
                        help='Threshold for adjacency matrix creation')
    parser.add_argument('--save-dir', type=str, default='./models',
                        help='Directory to save models')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of workers for data loading')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to use (for quick testing)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size for training')
    return parser.parse_args()

def train_model_on_voc(args):
    """
    Train ImageGCN model on VOC 2007 dataset
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get trainval dataset for training
    trainval_dataset = get_voc_dataset(
        args.voc_dir, 'trainval', 
        image_size=args.image_size, 
        max_images=args.max_images
    )
    
    # Check if test split exists
    test_split_exists = False
    test_split_file = os.path.join(args.voc_dir, 'ImageSets', 'Main', 'test.txt')
    voc2007_dir = os.path.join(args.voc_dir, 'VOC2007')
    if os.path.exists(voc2007_dir):
        test_split_file = os.path.join(voc2007_dir, 'ImageSets', 'Main', 'test.txt')
    
    if os.path.exists(test_split_file):
        test_split_exists = True
        print(f"Found test split file: {test_split_file}")
    else:
        print(f"Test split file not found: {test_split_file}")
        print("Will use a portion of the validation set as the test set")
    
    # Get test dataset if it exists, otherwise we'll split trainval further
    test_dataset = None
    if test_split_exists:
        test_dataset = get_voc_dataset(
            args.voc_dir, 'test', 
            image_size=args.image_size, 
            max_images=args.max_images
        )
    
    # Split trainval into train, validation (and test if needed)
    if test_split_exists:
        # Split trainval into train and validation sets (80/20)
        train_size = int(0.8 * len(trainval_dataset))
        val_size = len(trainval_dataset) - train_size
        train_dataset, val_dataset = random_split(
            trainval_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        # Split trainval into train, validation, and test sets (70/15/15)
        train_size = int(0.7 * len(trainval_dataset))
        val_size = int(0.15 * len(trainval_dataset))
        test_size = len(trainval_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            trainval_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    print(f"Train set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")
    print(f"Test set: {len(test_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    # Load all labels for creating adjacency matrix
    trainval_image_paths, trainval_labels, _ = load_voc_data(args.voc_dir, 'trainval')
    
    # Create adjacency matrix based on label co-occurrences
    print("Creating adjacency matrix from label co-occurrences...")
    adj_matrix = create_adjacency_matrix(trainval_labels, threshold=args.adj_threshold)
    
    # Initialize model
    feature_dim = 768 if args.use_vit else 2048  # ViT-Base vs ResNet-50
    model = ImageGCNSimple(
        num_classes=NUM_CLASSES,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_vit=args.use_vit,
        device=device
    )
    model.to(device)
    
    # Print model architecture and parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = ImageGCNTrainer(
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device
    )
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    best_val_f1 = 0.0
    
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss = trainer.train_epoch(train_loader)
        
        # Evaluate on validation set
        val_results = trainer.evaluate(val_loader)
        val_f1 = val_results['f1']
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val F1: {val_f1:.4f}, Precision: {val_results['precision']:.4f}, "
              f"Recall: {val_results['recall']:.4f}, Accuracy: {val_results['accuracy']:.4f}")
        
        # Save model if it's the best so far
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_path = os.path.join(args.save_dir, 'best_model_voc.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  New best model saved to {model_path}")
            
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_voc_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_val_f1': best_val_f1,
            }, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
    
    # Evaluate on test set using best model
    print("\nEvaluating best model on test set...")
    best_model_path = os.path.join(args.save_dir, 'best_model_voc.pth')
    model.load_state_dict(torch.load(best_model_path))
    
    test_results = trainer.evaluate(test_loader)
    print(f"Test Results:")
    print(f"  F1 Score: {test_results['f1']:.4f}")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall: {test_results['recall']:.4f}")
    print(f"  Accuracy: {test_results['accuracy']:.4f}")
    
    return model, test_results

def main():
    args = parse_args()
    train_model_on_voc(args)

if __name__ == "__main__":
    main() 