import os
import json
import argparse
import numpy as np
import pickle
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Generate adjacency matrix for COCO dataset')
    parser.add_argument('--data', type=str, required=True, help='Path to COCO dataset')
    parser.add_argument('--output', type=str, default='data/adj_matrix.pkl', help='Output path for adjacency matrix')
    parser.add_argument('--threshold', type=float, default=0.4, help='Threshold for adjacency matrix')
    parser.add_argument('--weight', type=float, default=0.2, help='Weight for adjacency matrix')
    return parser.parse_args()

def build_adjacency_matrix(labels, num_classes, tau=0.4, p=0.2):
    """
    Build adjacency matrix from co-occurrence statistics
    
    Args:
        labels: List of multi-hot label vectors
        num_classes: Number of classes
        tau: Threshold for co-occurrence probabilities
        p: Weight for re-weighting
        
    Returns:
        Adjacency matrix with self-connections
    """
    print("Building co-occurrence matrix...")
    M = np.zeros((num_classes, num_classes))
    N = np.zeros(num_classes)
    
    # Count occurrences and co-occurrences
    for label in tqdm(labels):
        # Get indices where label is 1
        indices = np.where(label == 1)[0]
        
        # Update counts
        for i in indices:
            N[i] += 1
            for j in indices:
                if i != j:
                    M[i, j] += 1
    
    # Compute conditional probabilities
    print("Computing conditional probabilities...")
    P = np.divide(M, N[:, None], out=np.zeros_like(M), where=N[:, None]!=0)
    
    # Binarize and re-weight
    print(f"Applying threshold {tau} and weight {p}...")
    A = (P >= tau).astype(np.float32)
    A_prime = np.zeros_like(A)
    
    for i in range(num_classes):
        row_sum = A[i].sum()
        if row_sum == 0:
            A_prime[i, i] = 1.0
        else:
            A_prime[i] = A[i] * (p / row_sum)
            A_prime[i, i] = 1 - p
    
    return A_prime

def process_coco(data_path):
    """Process COCO dataset and generate adjacency matrix"""
    # Load COCO annotations
    ann_file = os.path.join(data_path, "annotations", "instances_train2017.json")
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation file {ann_file} not found")
    
    print(f"Loading COCO annotations from {ann_file}")
    with open(ann_file, 'r') as f:
        dataset = json.load(f)
    
    # Create category id mapping
    categories = dataset['categories']
    num_classes = len(categories)
    coco_id_to_index = {cat['id']: i for i, cat in enumerate(categories)}
    
    # Create image dictionary for faster access
    img_dict = {img['id']: img for img in dataset['images']}
    
    # Group annotations by image
    img_to_anns = {}
    for ann in dataset['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Create multi-label vectors
    labels = []
    for img_id, anns in tqdm(img_to_anns.items(), desc="Processing annotations"):
        if img_id not in img_dict:
            continue
            
        # Create multi-label vector for this image
        target = np.zeros(num_classes, dtype=np.float32)
        for ann in anns:
            category_id = ann['category_id']
            if category_id in coco_id_to_index:
                label_idx = coco_id_to_index[category_id]
                target[label_idx] = 1.0
        
        labels.append(target)
    
    print(f"Created {len(labels)} multi-label vectors with {num_classes} categories")
    return np.array(labels), num_classes

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Process COCO dataset
    labels, num_classes = process_coco(args.data)
    
    # Build adjacency matrix
    adj_matrix = build_adjacency_matrix(labels, num_classes, args.threshold, args.weight)
    
    # Count occurrences per category
    nums = np.sum(labels, axis=0)
    
    # Save adjacency matrix and occurrence counts
    print(f"Saving adjacency matrix to {args.output}")
    result = {'adj': adj_matrix, 'nums': nums}
    with open(args.output, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print("Done!")

if __name__ == "__main__":
    main() 