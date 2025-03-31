import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from gcn_image_tagger_simplified import ImageGCNSimple, create_adjacency_matrix, ImageTaggingDataset, ImageGCNTrainer

# COCO dataset constants
NUM_CLASSES = 80  # COCO has 80 categories

def load_coco_data(coco_dir, split='val2017'):
    """
    Load COCO dataset images and annotations
    
    Args:
        coco_dir: Path to COCO dataset directory
        split: Dataset split ('train2017', 'val2017', 'test2017')
        
    Returns:
        image_paths: List of image paths
        labels: Numpy array of labels (one-hot encoded)
        category_names: List of category names
    """
    # Try multiple possible directory structures
    possible_img_dirs = [
        os.path.join(coco_dir, split),                        # Kaggle: coco_dir/val2017
        os.path.join(coco_dir, 'coco2017', split),            # Kaggle nested: coco_dir/coco2017/val2017
        os.path.join(coco_dir, 'images', split),              # Standard: coco_dir/images/val2017
        os.path.join(coco_dir),                               # All images in root
    ]
    
    # Print possible directories for debugging
    print(f"Looking for images in possible locations:")
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
                    print(f"Found images in: {img_dir}")
                    break
            except Exception as e:
                print(f"Error listing {dir_path}: {e}")
    
    if img_dir is None:
        raise FileNotFoundError(f"Could not find image directory for {split} in {coco_dir}")
    
    # Try multiple possible annotation file locations
    possible_ann_files = [
        os.path.join(coco_dir, 'annotations', f'instances_{split}.json'),  # Standard: coco_dir/annotations
        os.path.join(coco_dir, 'coco2017', 'annotations', f'instances_{split}.json'), # Nested
        os.path.join(coco_dir, f'instances_{split}.json'),                  # Root level
    ]
    
    # Print possible annotation files for debugging
    print(f"Looking for annotation files in possible locations:")
    for file_path in possible_ann_files:
        print(f"  - {file_path}")
    
    # Find the first valid annotation file
    ann_file = None
    for file_path in possible_ann_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            ann_file = file_path
            print(f"Found annotations in: {ann_file}")
            break
    
    if ann_file is None:
        raise FileNotFoundError(f"Could not find annotation file for {split} in {coco_dir}")
    
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
    
    for img_id, anns in tqdm(image_annotations.items(), desc=f"Processing {split}"):
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
    
    print(f"Loaded {len(image_paths)} images with {NUM_CLASSES} categories")
    return image_paths, np.array(labels), category_names

def evaluate_model_on_coco(coco_dir, model_path=None, batch_size=8, device='cuda', max_images=5000, image_size=128):
    """
    Evaluate ImageGCN model on COCO dataset
    
    Args:
        coco_dir: Path to COCO dataset directory
        model_path: Path to saved model weights (optional)
        batch_size: Batch size for evaluation
        device: Device to use (cuda/cpu)
        max_images: Maximum number of images to use (for quicker evaluation)
        image_size: Image size for evaluation
        
    Returns:
        evaluation_results: Dictionary with evaluation metrics
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load validation data
    val_image_paths, val_labels, category_names = load_coco_data(coco_dir, split='val2017')
    
    # Limit number of images if specified
    if max_images and len(val_image_paths) > max_images:
        print(f"Limiting evaluation to {max_images} images for faster processing")
        indices = np.random.choice(len(val_image_paths), max_images, replace=False)
        val_image_paths = [val_image_paths[i] for i in indices]
        val_labels = val_labels[indices]
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and data loader
    val_dataset = ImageTaggingDataset(val_image_paths, val_labels, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create adjacency matrix from labels
    adj_matrix = create_adjacency_matrix(val_labels, threshold=0.2)
    
    # Initialize model
    model = ImageGCNSimple(
        num_classes=NUM_CLASSES,
        feature_dim=2048,  # ResNet-50
        hidden_dim=512,
        dropout=0.5,
        use_vit=False,  # Use ResNet-50
        device=device
    )
    
    # Load model weights if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No model weights provided. Using randomly initialized model.")
    
    # Initialize trainer
    trainer = ImageGCNTrainer(model, device=device)
    
    # Evaluate model
    print("Evaluating model on COCO val2017 dataset...")
    
    # Collect predictions and ground truth for calculating mAP
    all_preds = []
    all_probs = []  # Store probability scores
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate results
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics
    accuracy = np.mean((all_preds == all_labels).flatten())
    
    # Per-class metrics
    # Avoid division by zero
    class_correct = np.sum(all_preds & (all_labels == 1), axis=0)
    class_total_pred = np.sum(all_preds, axis=0)
    class_total_true = np.sum(all_labels, axis=0)
    
    # Add epsilon to avoid division by zero
    epsilon = 1e-10
    precision = class_correct / (class_total_pred + epsilon)
    recall = class_correct / (class_total_true + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    
    # Calculate mAP (mean Average Precision)
    # For each class, compute average precision
    ap_scores = []
    for i in range(NUM_CLASSES):
        if np.sum(all_labels[:, i]) > 0:  # Only calculate AP if class exists in test set
            ap = average_precision_score(all_labels[:, i], all_probs[:, i])
            ap_scores.append(ap)
    
    mAP = np.mean(ap_scores)
    
    # Collect results
    results = {
        'accuracy': accuracy,
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1': np.mean(f1),
        'mAP': mAP,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_ap': {category_names[i]: ap_scores[i] for i in range(len(ap_scores)) if np.sum(all_labels[:, i]) > 0}
    }
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"mAP: {mAP:.4f}")
    
    # Print per-class results for top and bottom 5 classes
    per_class_f1 = results['per_class_f1']
    indices = np.argsort(per_class_f1)
    
    print("\nTop 5 classes:")
    for i in indices[-5:]:
        print(f"  {category_names[i]}: F1={per_class_f1[i]:.4f}, "
              f"Precision={results['per_class_precision'][i]:.4f}, "
              f"Recall={results['per_class_recall'][i]:.4f}")
    
    print("\nBottom 5 classes:")
    for i in indices[:5]:
        print(f"  {category_names[i]}: F1={per_class_f1[i]:.4f}, "
              f"Precision={results['per_class_precision'][i]:.4f}, "
              f"Recall={results['per_class_recall'][i]:.4f}")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate ImageGCN on COCO dataset (Simplified Version)')
    parser.add_argument('--coco-dir', type=str, required=True, 
                        help='Path to COCO dataset directory')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved model weights')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--max-images', type=int, default=5000,
                        help='Maximum number of images to use (for quicker evaluation)')
    parser.add_argument('--image-size', type=int, default=128,
                        help='Image size for evaluation')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    try:
        evaluate_model_on_coco(
            coco_dir=args.coco_dir,
            model_path=args.model_path,
            batch_size=args.batch_size,
            device=args.device,
            max_images=args.max_images,
            image_size=args.image_size
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPossible solutions:")
        print("1. Check if your COCO dataset path is correct")
        print("2. Make sure annotations/instances_val2017.json exists")
        print("3. Verify that dataset contains images in expected locations")
        print("\nCOCO dataset structure tips:")
        print("- Standard structure: coco_dir/images/val2017/")
        print("- Kaggle structure: coco_dir/val2017/")
        print("- Annotations should be in coco_dir/annotations/")
        print("\nExample usage:")
        print("python evaluate_coco_simplified.py --coco-dir /path/to/coco --model-path ./models/best_model.pth --max-images 1000")

if __name__ == "__main__":
    main() 