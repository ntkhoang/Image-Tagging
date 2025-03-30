import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from gcn_image_tagger_simplified import ImageGCNSimple, create_adjacency_matrix, ImageTaggingDataset, ImageGCNTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train ImageGCN model for image tagging (Simplified Version)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
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
    parser.add_argument('--image-size', type=int, default=128,
                        help='Image size for training (smaller is faster)')
    return parser.parse_args()

def load_images_and_labels(data_dir):
    """
    Load images and labels from a directory
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        image_paths: List of image paths
        labels: Numpy array of labels (one-hot encoded)
    """
    import glob
    import json
    from collections import defaultdict
    
    # Check if this appears to be a COCO-style dataset
    ann_dir = os.path.join(data_dir, 'annotations')
    img_dir = os.path.join(data_dir, 'images')
    
    # Try direct structure first (if directories are directly in data_dir)
    is_coco_structure = os.path.isdir(ann_dir) and os.path.exists(img_dir)
    
    # If not found, check if data_dir itself is the root containing annotations and split folders
    if not is_coco_structure:
        ann_dir = os.path.join(data_dir, 'annotations')
        # COCO might not have an 'images' directory, but instead have train2017, val2017 etc. directly
        is_coco_structure = (os.path.isdir(ann_dir) and 
                           (os.path.isdir(os.path.join(data_dir, 'train2017')) or
                            os.path.isdir(os.path.join(data_dir, 'val2017'))))
        img_dir = data_dir  # In this case, splits are directly in data_dir
    
    if is_coco_structure:
        print("Detected COCO-style dataset structure")
        # Find the annotation files
        ann_files = glob.glob(os.path.join(ann_dir, "instances_*.json"))
        if not ann_files:
            raise FileNotFoundError(f"No annotation files found in {ann_dir}")
        
        # Prefer train2017 for training
        train_ann_file = None
        for ann_file in ann_files:
            if 'train' in os.path.basename(ann_file):
                train_ann_file = ann_file
                break
        
        # If no train annotation found, use the first available
        ann_file = train_ann_file or ann_files[0]
        print(f"Loading annotations from {ann_file}")
        
        # Load the annotation file
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Get category information
        categories = {cat['id']: i for i, cat in enumerate(data['categories'])}
        category_names = [cat['name'] for cat in sorted(data['categories'], key=lambda x: categories[x['id']])]
        num_classes = len(categories)
        
        # Create image id to filename mapping
        image_dict = {img['id']: img['file_name'] for img in data['images']}
        
        # Group annotations by image
        image_annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # Determine which split to use based on the annotation file
        split = os.path.basename(ann_file).replace('instances_', '').replace('.json', '')
        split_dir = os.path.join(img_dir, split)
        
        # If specific split directory doesn't exist, try using any image subdirectory
        if not os.path.isdir(split_dir):
            img_subdirs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d)) 
                           and not d == 'annotations']
            if img_subdirs:
                split_dir = os.path.join(img_dir, img_subdirs[0])
            else:
                split_dir = img_dir  # Fallback to using the main image directory
        
        # Create dataset
        image_paths = []
        labels = []
        
        # Process each image with annotations
        print(f"Processing {len(image_annotations)} images with annotations...")
        for img_id, anns in image_annotations.items():
            if img_id not in image_dict:
                continue
            
            file_name = image_dict[img_id]
            img_path = os.path.join(split_dir, file_name)
            
            # Skip if image doesn't exist
            if not os.path.exists(img_path):
                # Try other possible split directories
                found = False
                for possible_dir in [os.path.join(img_dir, d) for d in os.listdir(img_dir) 
                                    if os.path.isdir(os.path.join(img_dir, d)) and d != 'annotations']:
                    possible_path = os.path.join(possible_dir, file_name)
                    if os.path.exists(possible_path):
                        img_path = possible_path
                        found = True
                        break
                
                if not found:
                    continue
            
            # Create one-hot label vector
            label = np.zeros(num_classes)
            for ann in anns:
                category_id = ann['category_id']
                if category_id in categories:
                    label_idx = categories[category_id]
                    label[label_idx] = 1
            
            image_paths.append(img_path)
            labels.append(label)
        
        if len(image_paths) == 0:
            raise ValueError(f"No valid images found with annotations in {split_dir}. Please check your dataset.")
            
        labels = np.array(labels)
        print(f"Loaded {len(image_paths)} images with {num_classes} classes")
        print(f"Class names: {category_names[:10]}..." if len(category_names) > 10 else f"Class names: {category_names}")
        
    # Example: scan for a labels file
    elif os.path.exists(os.path.join(data_dir, 'labels.txt')):
        labels_file = os.path.join(data_dir, 'labels.txt')
        print(f"Loading labels from {labels_file}")
        
        # Example format: image_path,label1,label2,...
        image_paths = []
        all_labels = []
        unique_labels = set()
        
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue
                
                img_path = os.path.join(data_dir, parts[0])
                if not os.path.exists(img_path):
                    continue
                
                image_paths.append(img_path)
                img_labels = parts[1:]
                all_labels.append(img_labels)
                
                for lbl in img_labels:
                    unique_labels.add(lbl)
        
        # Create label mapping
        label_to_idx = {label: i for i, label in enumerate(sorted(unique_labels))}
        num_classes = len(label_to_idx)
        print(f"Found {num_classes} unique classes")
        
        # Convert text labels to one-hot encoded vectors
        labels = np.zeros((len(all_labels), num_classes))
        for i, img_labels in enumerate(all_labels):
            for lbl in img_labels:
                labels[i, label_to_idx[lbl]] = 1
    else:
        # Alternative: scan for images and derive labels from directory structure
        # Example: data_dir/class1/img1.jpg, data_dir/class2/img2.jpg, etc.
        class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"Found {len(class_dirs)} classes: {class_dirs}")
        
        # Filter out directories that aren't actually classes (like 'annotations' or 'images')
        image_paths = []
        all_labels = []
        
        # Check if any of the directories contain images
        valid_class_dirs = []
        for class_dir in class_dirs:
            class_path = os.path.join(data_dir, class_dir)
            class_images = glob.glob(os.path.join(class_path, "*.jpg")) + \
                          glob.glob(os.path.join(class_path, "*.jpeg")) + \
                          glob.glob(os.path.join(class_path, "*.png"))
            if class_images:
                valid_class_dirs.append(class_dir)
        
        if valid_class_dirs:
            label_to_idx = {label: i for i, label in enumerate(sorted(valid_class_dirs))}
            num_classes = len(label_to_idx)
            
            for class_name in valid_class_dirs:
                class_path = os.path.join(data_dir, class_name)
                class_images = glob.glob(os.path.join(class_path, "*.jpg")) + \
                              glob.glob(os.path.join(class_path, "*.jpeg")) + \
                              glob.glob(os.path.join(class_path, "*.png"))
                
                for img_path in class_images:
                    image_paths.append(img_path)
                    
                    # Single-label case (convert to one-hot)
                    label_vec = np.zeros(num_classes)
                    label_vec[label_to_idx[class_name]] = 1
                    all_labels.append(label_vec)
            
            labels = np.array(all_labels)
            print(f"Loaded {len(image_paths)} images with {num_classes} classes")
        else:
            raise ValueError("Could not find any valid image classes in the directory structure. "
                             "Please check your data directory organization.")
    
    if len(image_paths) == 0:
        raise ValueError("No images found! Please check your data directory.")
        
    return image_paths, labels

def train_model(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load images and labels
    image_paths, labels = load_images_and_labels(args.data_dir)
    
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
    
    # Create transforms with smaller image size
    img_size = args.image_size
    print(f"Using image size: {img_size}x{img_size}")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # Create adjacency matrix from training labels
    # Note: This adjacency matrix is not used in the simplified implementation,
    # but we keep it for compatibility with the original code
    adj_matrix = create_adjacency_matrix(train_labels, threshold=args.adj_threshold)
    
    # Get number of classes from labels shape
    num_classes = labels.shape[1]
    
    # Initialize model
    feature_dim = 768 if args.use_vit else 2048  # ViT-Base vs ResNet-50
    model = ImageGCNSimple(
        num_classes=num_classes,
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
    print(f"Model Architecture: {model}")
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
            model_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  New best model saved to {model_path}")
            
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_val_f1': best_val_f1,
            }, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
    
    # Evaluate on test set using best model
    print("\nEvaluating best model on test set...")
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
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
    train_model(args)

if __name__ == "__main__":
    main() 