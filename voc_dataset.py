import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# VOC 2007 classes
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

NUM_CLASSES = len(VOC_CLASSES)
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}

def load_voc_data(voc_dir, split='val'):
    """
    Load VOC 2007 dataset images and annotations
    
    Args:
        voc_dir: Path to VOC2007 dataset directory
        split: Dataset split ('train', 'val', 'test')
        
    Returns:
        image_paths: List of image paths
        labels: Numpy array of labels (one-hot encoded)
        class_names: List of class names
    """
    # Map splits to VOC-specific folder names
    if split == 'train':
        split_name = 'train'
    elif split == 'val':
        split_name = 'val'
    elif split == 'test':
        split_name = 'test'
    else:
        split_name = 'trainval'  # Default to trainval
    
    # Check for VOC2007 structure
    img_dir = os.path.join(voc_dir, 'JPEGImages')
    ann_dir = os.path.join(voc_dir, 'Annotations')
    split_file = os.path.join(voc_dir, 'ImageSets', 'Main', f'{split_name}.txt')
    
    # If directories don't exist, try looking for a specific VOC2007 subdirectory
    if not os.path.exists(img_dir):
        voc2007_dir = os.path.join(voc_dir, 'VOC2007')
        if os.path.exists(voc2007_dir):
            img_dir = os.path.join(voc2007_dir, 'JPEGImages')
            ann_dir = os.path.join(voc2007_dir, 'Annotations')
            split_file = os.path.join(voc2007_dir, 'ImageSets', 'Main', f'{split_name}.txt')
    
    print(f"Looking for VOC data in:")
    print(f"  Images: {img_dir}")
    print(f"  Annotations: {ann_dir}")
    print(f"  Split file: {split_file}")
    
    # Validate directories exist
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    # Read image IDs from split file
    with open(split_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]
    
    image_paths = []
    labels = []
    
    # Process each image
    print(f"Processing {len(image_ids)} images from {split_name} split...")
    for img_id in image_ids:
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        ann_path = os.path.join(ann_dir, f"{img_id}.xml")
        
        # Skip if files don't exist
        if not os.path.exists(img_path) or not os.path.exists(ann_path):
            continue
        
        # Parse XML annotation
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        # Create one-hot label vector
        label = np.zeros(NUM_CLASSES)
        for obj in root.findall('object'):
            cls_name = obj.find('name').text.lower().strip()
            if cls_name in CLASS_TO_IDX:
                label_idx = CLASS_TO_IDX[cls_name]
                label[label_idx] = 1
        
        image_paths.append(img_path)
        labels.append(label)
    
    print(f"Loaded {len(image_paths)} images with {NUM_CLASSES} classes")
    return image_paths, np.array(labels), VOC_CLASSES

class VOCDataset(Dataset):
    """
    Dataset for Pascal VOC 2007
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a blank image as fallback
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, torch.tensor(label, dtype=torch.float32)

def get_voc_dataset(voc_dir, split='train', image_size=224, max_images=None):
    """
    Helper function to create a VOC dataset with proper transforms
    
    Args:
        voc_dir: Path to VOC2007 dataset directory
        split: Dataset split ('train', 'val', 'test')
        image_size: Size of images
        max_images: Maximum number of images to use (optional)
        
    Returns:
        dataset: VOCDataset object
    """
    # Load data
    image_paths, labels, class_names = load_voc_data(voc_dir, split)
    
    # Limit number of images if specified
    if max_images and len(image_paths) > max_images:
        indices = np.random.choice(len(image_paths), max_images, replace=False)
        image_paths = [image_paths[i] for i in indices]
        labels = labels[indices]
    
    # Create transforms
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create dataset
    dataset = VOCDataset(image_paths, labels, transform=transform)
    
    return dataset 