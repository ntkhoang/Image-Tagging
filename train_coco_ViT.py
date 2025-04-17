import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import glob

from gcn_image_tagger_simplified import ImageGCNSimple, create_adjacency_matrix, ImageTaggingDataset, ImageGCNTrainer

# COCO dataset constants
NUM_CLASSES = 80  # COCO has 80 categories

def parse_args():
    parser = argparse.ArgumentParser(description='Train ImageGCN model on COCO dataset with Vision Transformer backbone')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to COCO dataset directory')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for training')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Hidden dimension for GCN layers')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate (reduced for ViT)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (adjusted for ViT)')
    parser.add_argument('--adj-threshold', type=float, default=0.2,
                        help='Threshold for adjacency matrix creation')
    parser.add_argument('--save-dir', type=str, default='./models_vit',
                        help='Directory to save models')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of workers for data loading')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to use (for quick testing)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size for training (can use 224, 384, 448, etc. for ViT)')
    parser.add_argument('--train-backbone', action='store_true',
                        help='Train ViT backbone parameters (increases GPU memory usage)')
    return parser.parse_args()

def load_coco_data(coco_dir):
    """
    Load COCO dataset images and annotations
    
    Args:
        coco_dir: Path to COCO dataset directory
        
    Returns:
        image_paths: List of image paths
        labels: Numpy array of labels (one-hot encoded)
        category_names: List of category names
    """
    # Try multiple possible directory structures
    possible_img_dirs = [
        os.path.join(coco_dir, 'train2017'),                   # Kaggle: coco_dir/train2017
        os.path.join(coco_dir, 'coco2017', 'train2017'),       # Kaggle nested: coco_dir/coco2017/train2017
        os.path.join(coco_dir, 'images', 'train2017'),         # Standard: coco_dir/images/train2017
        os.path.join(coco_dir),                                # All images in root
    ]
    
    # Print possible directories for debugging
    print(f"Looking for COCO train images in possible locations:")
    for dir_path in possible_img_dirs:
        print(f"  - {dir_path}")
    
    # Find the first valid image directory
    img_dir = None
    for dir_path in possible_img_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            # Check if directory contains images
            try:
                files = os.listdir(dir_path)[:100]
                if any(f.endswith('.jpg') for f in files if os.path.isfile(os.path.join(dir_path, f))):
                    img_dir = dir_path
                    print(f"Found train images in: {img_dir}")
                    break
            except Exception as e:
                print(f"Error listing {dir_path}: {e}")
    
    if img_dir is None:
        raise FileNotFoundError(f"Could not find train image directory in {coco_dir}")
    
    # Try multiple possible annotation file locations
    possible_ann_files = [
        os.path.join(coco_dir, 'annotations', 'instances_train2017.json'),  # Standard: coco_dir/annotations
        os.path.join(coco_dir, 'coco2017', 'annotations', 'instances_train2017.json'), # Nested
        os.path.join(coco_dir, 'instances_train2017.json'),                  # Root level
    ]
    
    # Print possible annotation files for debugging
    print(f"Looking for COCO train annotation files in possible locations:")
    for file_path in possible_ann_files:
        print(f"  - {file_path}")
    
    # Find the first valid annotation file
    ann_file = None
    for file_path in possible_ann_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            ann_file = file_path
            print(f"Found train annotations in: {ann_file}")
            break
    
    if ann_file is None:
        raise FileNotFoundError(f"Could not find annotation file for train2017 in {coco_dir}")
    
    print(f"Loading COCO annotations from {ann_file}")
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Get category information
    categories = {cat['id']: i for i, cat in enumerate(data['categories'])}
    category_names = [cat['name'] for cat in sorted(data['categories'], key=lambda x: categories[x['id']])]
    
    # Create image id to filename mapping
    image_dict = {img['id']: img['file_name'] for img in data['images']}
    
    # Group annotations by image
    image_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Create dataset
    image_paths = []
    labels = []
    
    # Process each image with annotations
    print(f"Processing {len(image_annotations)} images with annotations...")
    for img_id, anns in tqdm(image_annotations.items(), desc="Processing train images"):
        if img_id not in image_dict:
            continue
        
        file_name = image_dict[img_id]
        img_path = os.path.join(img_dir, file_name)
        
        # Skip if image doesn't exist
        if not os.path.exists(img_path):
            continue
        
        # Create one-hot label vector
        label = np.zeros(NUM_CLASSES)
        for ann in anns:
            category_id = ann['category_id']
            if category_id in categories:
                label_idx = categories[category_id]
                label[label_idx] = 1
        
        image_paths.append(img_path)
        labels.append(label)
    
    print(f"Loaded {len(image_paths)} train images with {NUM_CLASSES} categories")
    return image_paths, np.array(labels), category_names

def train_model_on_coco_vit(args):
    """
    Train ImageGCN model on COCO dataset with Vision Transformer backbone
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check for multiple GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    use_multi_gpu = num_gpus > 1 and args.device == 'cuda'
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load images and labels
    image_paths, labels, category_names = load_coco_data(args.data_dir)
    
    # Limit number of images for quick testing if specified
    if args.max_images and args.max_images < len(image_paths):
        print(f"Limiting to {args.max_images} images for faster training")
        indices = np.random.choice(len(image_paths), args.max_images, replace=False)
        image_paths = [image_paths[i] for i in indices]
        labels = labels[indices]
    
    # Split dataset into train, validation, and test sets
    train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels.sum(axis=1) > 0
    )
    
    val_img_paths, test_img_paths, val_labels, test_labels = train_test_split(
        val_img_paths, val_labels, test_size=0.5, random_state=42, stratify=val_labels.sum(axis=1) > 0
    )
    
    print(f"Train set: {len(train_img_paths)} images")
    print(f"Validation set: {len(val_img_paths)} images")
    print(f"Test set: {len(test_img_paths)} images")
    
    # Create transforms with appropriate image size for ViT
    img_size = args.image_size
    print(f"Using image size: {img_size}x{img_size}")
    
    # Add custom ViT model for non-standard sizes
    if img_size != 224:
        # If image size isn't the standard 224, we need to modify ViT loading in ImageGCNSimple
        # We'll need to modify our model to handle this
        print(f"Using custom Vision Transformer with image size {img_size}x{img_size}")
        print(f"Note: Using a non-standard image size with ViT requires position embedding interpolation")
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and data loaders
    train_dataset = ImageTaggingDataset(train_img_paths, train_labels, transform=train_transform)
    val_dataset = ImageTaggingDataset(val_img_paths, val_labels, transform=val_transform)
    test_dataset = ImageTaggingDataset(test_img_paths, test_labels, transform=val_transform)
    
    # Adjust batch size for multi-GPU training
    effective_batch_size = args.batch_size
    if use_multi_gpu:
        effective_batch_size = args.batch_size * num_gpus
        print(f"Using effective batch size of {effective_batch_size} with {num_gpus} GPUs")
    
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=effective_batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # Create adjacency matrix from training labels
    print("Creating adjacency matrix from training labels...")
    adj_matrix = create_adjacency_matrix(train_labels, threshold=args.adj_threshold)
    
    # Initialize model - always use ViT as backbone
    feature_dim = 768  # ViT-Base feature dimension
    model = ImageGCNSimple(
        num_classes=NUM_CLASSES,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_vit=True,  # Force ViT usage
        device=str(device),
        train_backbone=args.train_backbone,
        img_size=img_size  # Pass the image size
    )
    model.to(device)
    
    # Wrap model with DataParallel if multiple GPUs are available
    if use_multi_gpu:
        print("Using DataParallel for multi-GPU training")
        model = torch.nn.DataParallel(model)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Using Vision Transformer (ViT) as the backbone")
    
    # Initialize trainer
    trainer = ImageGCNTrainer(
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=str(device)
    )
    
    # Train model
    print("Starting training...")
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
            model_path = os.path.join(args.save_dir, 'best_model_coco_vit.pth')
            # Save the model state dict, handling DataParallel case
            if use_multi_gpu:
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            print(f"  New best model saved to {model_path}")
            
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_coco_vit_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if use_multi_gpu else model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_val_f1': best_val_f1,
            }, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
    
    # Evaluate on test set using best model
    print("\nEvaluating best model on test set...")
    best_model_path = os.path.join(args.save_dir, 'best_model_coco_vit.pth')
    
    # Reload model for evaluation
    if use_multi_gpu:
        model.module.load_state_dict(torch.load(best_model_path))
    else:
        model.load_state_dict(torch.load(best_model_path))
    
    test_results = trainer.evaluate(test_loader)
    print(f"Test Results:")
    print(f"  F1 Score: {test_results['f1']:.4f}")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall: {test_results['recall']:.4f}")
    print(f"  Accuracy: {test_results['accuracy']:.4f}")
    
    # Print per-class results for top and bottom 5 classes
    per_class_f1 = test_results['per_class_f1']
    indices = np.argsort(per_class_f1)
    
    print("\nTop 5 classes by F1 score:")
    for i in indices[-5:]:
        print(f"  {category_names[i]}: F1={per_class_f1[i]:.4f}")
    
    print("\nBottom 5 classes by F1 score:")
    for i in indices[:5]:
        print(f"  {category_names[i]}: F1={per_class_f1[i]:.4f}")
    
    return model, test_results

def main():
    args = parse_args()
    train_model_on_coco_vit(args)

if __name__ == "__main__":
    main() 