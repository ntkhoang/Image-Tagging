import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

from voc_dataset import get_voc_dataset, load_voc_data, VOC_CLASSES, NUM_CLASSES
from gcn_image_tagger_simplified import ImageGCNSimple

def evaluate_model_on_voc(voc_dir, model_path=None, batch_size=16, 
                         device='cuda', max_images=None, image_size=224,
                         num_workers=2, visualize=False, split='test'):
    """
    Evaluate a trained ImageGCN model on VOC 2007 test set
    
    Args:
        voc_dir: Path to VOC dataset directory
        model_path: Path to saved model weights
        batch_size: Batch size for evaluation
        device: Device to use (cuda/cpu)
        max_images: Maximum number of images to evaluate
        image_size: Size of images for evaluation
        num_workers: Number of workers for data loading
        visualize: Whether to visualize results
        split: Dataset split to evaluate on ('test', 'val', 'trainval')
        
    Returns:
        results: Dictionary with evaluation metrics
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if test split exists
    test_split_exists = False
    test_split_file = os.path.join(voc_dir, 'ImageSets', 'Main', 'test.txt')
    voc2007_dir = os.path.join(voc_dir, 'VOC2007')
    if os.path.exists(voc2007_dir):
        test_split_file = os.path.join(voc2007_dir, 'ImageSets', 'Main', 'test.txt')
    
    if os.path.exists(test_split_file):
        test_split_exists = True
        print(f"Found test split file: {test_split_file}")
    else:
        print(f"Test split file not found: {test_split_file}")
        if split == 'test':
            print("Falling back to 'val' split for evaluation")
            split = 'val'
    
    # Get dataset for evaluation
    try:
        eval_dataset = get_voc_dataset(
            voc_dir, split, 
            image_size=image_size, 
            max_images=max_images
        )
    except FileNotFoundError:
        # If specified split is not found, try falling back to trainval and creating a split
        if split in ['test', 'val']:
            print(f"Split '{split}' not found, falling back to splitting trainval")
            trainval_dataset = get_voc_dataset(
                voc_dir, 'trainval', 
                image_size=image_size, 
                max_images=max_images
            )
            
            # Split the trainval dataset (70/15/15)
            train_size = int(0.7 * len(trainval_dataset))
            val_size = int(0.15 * len(trainval_dataset))
            test_size = len(trainval_dataset) - train_size - val_size
            
            if split == 'val':
                # Use the validation portion
                _, eval_split, _ = random_split(
                    trainval_dataset, 
                    [train_size, val_size, test_size],
                    generator=torch.Generator().manual_seed(42)
                )
                eval_dataset = eval_split
            else:  # split == 'test'
                # Use the test portion
                _, _, eval_split = random_split(
                    trainval_dataset, 
                    [train_size, val_size, test_size],
                    generator=torch.Generator().manual_seed(42)
                )
                eval_dataset = eval_split
        else:
            raise
    
    # Create data loader
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Evaluating on {len(eval_dataset)} {split} images...")
    
    # Initialize model
    model = ImageGCNSimple(
        num_classes=NUM_CLASSES,
        feature_dim=2048,  # ResNet by default
        hidden_dim=512,
        dropout=0.0,  # No dropout for evaluation
        device=device
    )
    
    # Load model weights if provided
    if model_path:
        print(f"Loading model weights from {model_path}")
        try:
            # First try loading as is (normal model save)
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            
            # Check if this is a checkpoint file with multiple dictionaries
            if 'model_state_dict' in state_dict:
                print("Detected checkpoint file, extracting model state dictionary")
                state_dict = state_dict['model_state_dict']
            
            # Check if the state_dict has DataParallel prefix 'module.' in keys
            is_data_parallel = False
            if len(list(state_dict.keys())) > 0:
                is_data_parallel = list(state_dict.keys())[0].startswith('module.')
                
            if is_data_parallel:
                print("Detected DataParallel model, removing 'module.' prefix")
                # Create new OrderedDict without the 'module.' prefix
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
                    new_state_dict[name] = v
                state_dict = new_state_dict
                
            model.load_state_dict(state_dict)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Continuing with randomly initialized weights...")
    else:
        print("No model weights provided. Using randomly initialized model.")
    
    model.to(device)
    model.eval()
    
    # Collect predictions and ground truth
    all_preds = []
    all_probs = []  # Store probability scores
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(eval_loader, desc="Evaluating"):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
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
        'per_class_ap': {VOC_CLASSES[i]: ap_scores[i] for i in range(len(ap_scores))}
    }
    
    # Print results
    print(f"\nEvaluation Results on {split} set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Precision: {np.mean(precision):.4f}")
    print(f"Mean Recall: {np.mean(recall):.4f}")
    print(f"Mean F1 Score: {np.mean(f1):.4f}")
    print(f"mAP: {mAP:.4f}")
    
    # Print per-class results (top 5 and bottom 5)
    indices = np.argsort(f1)
    
    print("\nTop 5 classes by F1 score:")
    for i in indices[-5:]:
        print(f"  {VOC_CLASSES[i]}: F1={f1[i]:.4f}, "
              f"Precision={precision[i]:.4f}, Recall={recall[i]:.4f}")
    
    print("\nBottom 5 classes by F1 score:")
    for i in indices[:5]:
        print(f"  {VOC_CLASSES[i]}: F1={f1[i]:.4f}, "
              f"Precision={precision[i]:.4f}, Recall={recall[i]:.4f}")
    
    # Visualize results if requested
    if visualize:
        # Plot per-class metrics
        plt.figure(figsize=(15, 10))
        plt.bar(VOC_CLASSES, precision, alpha=0.7, label='Precision')
        plt.bar(VOC_CLASSES, recall, alpha=0.7, label='Recall')
        plt.bar(VOC_CLASSES, f1, alpha=0.7, label='F1')
        plt.xticks(rotation=90)
        plt.ylabel('Score')
        plt.title(f'Per-class Performance on {split} set')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'voc_{split}_class_metrics.png')
        print(f"Per-class metrics visualization saved to voc_{split}_class_metrics.png")
        
        # Plot confusion matrix or distribution
        plt.figure(figsize=(10, 8))
        plt.matshow(np.corrcoef(all_labels.T), cmap='coolwarm')
        plt.colorbar()
        plt.title('Label Co-occurrence Matrix')
        plt.savefig(f'voc_{split}_label_cooccurrence.png')
        print(f"Label co-occurrence matrix saved to voc_{split}_label_cooccurrence.png")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate ImageGCN model on VOC 2007 dataset')
    parser.add_argument('--voc-dir', type=str, required=True,
                        help='Path to VOC dataset directory')
    parser.add_argument('--model-path', type=str,
                        help='Path to saved model weights')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to evaluate')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size for evaluation')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of workers for data loading')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize evaluation results')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate on (test, val, trainval)')
    
    args = parser.parse_args()
    
    try:
        evaluate_model_on_voc(
            voc_dir=args.voc_dir,
            model_path=args.model_path,
            batch_size=args.batch_size,
            device=args.device,
            max_images=args.max_images,
            image_size=args.image_size,
            num_workers=args.num_workers,
            visualize=args.visualize,
            split=args.split
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPossible solutions:")
        print("1. Check if your VOC dataset path is correct")
        print("2. Make sure the dataset has the expected directory structure")
        print("\nVOC dataset structure:")
        print("- JPEGImages/ - Directory with images")
        print("- Annotations/ - Directory with XML annotations")
        print("- ImageSets/Main/ - Directory with train/val/test splits")
        print("\nExample usage:")
        print("python evaluate_voc.py --voc-dir /path/to/VOCdevkit/VOC2007 --model-path ./models/best_model_voc.pth --split val")

if __name__ == "__main__":
    main() 