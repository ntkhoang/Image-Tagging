import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader

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
    img_dir = os.path.join(coco_dir, 'images', split)
    ann_file = os.path.join(coco_dir, 'annotations', f'instances_{split}.json')
    
    # Check if files exist
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    
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

def evaluate_model_on_coco(coco_dir, model_path=None, batch_size=16, device='cuda'):
    """
    Evaluate ImageGCN model on COCO dataset
    
    Args:
        coco_dir: Path to COCO dataset directory
        model_path: Path to saved model weights (optional)
        batch_size: Batch size for evaluation
        device: Device to use (cuda/cpu)
        
    Returns:
        evaluation_results: Dictionary with evaluation metrics
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load validation data
    val_image_paths, val_labels, category_names = load_coco_data(coco_dir, split='val2017')
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and data loader
    val_dataset = ImageTaggingDataset(val_image_paths, val_labels, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
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
    results = trainer.evaluate(val_loader, adj_matrix)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    
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
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    evaluate_model_on_coco(
        coco_dir=args.coco_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == "__main__":
    main() 