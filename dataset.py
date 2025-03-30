import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import json

class COCODataset(data.Dataset):
    def __init__(self, data_path, split='train', transform=None, target_transform=None):
        self.data_path = data_path
        self.split = split  # 'train' or 'val'
        self.transform = transform
        self.target_transform = target_transform
        
        # COCO paths
        self.img_dir = os.path.join(data_path, f"{split}2017")
        self.ann_file = os.path.join(data_path, "annotations", f"instances_{split}2017.json")
        
        # Load COCO annotations
        self.imgs = []
        self.labels = []
        self.coco_ids = []
        self.num_classes = 80
        
        self._load_coco()
        
        # Generate dummy data if no real data was loaded
        if len(self.imgs) == 0:
            print(f"Warning: No images found for {split}. Creating dummy data.")
            self._create_dummy_data()
    
        # Load word embeddings for GCN
        self.word_embeddings = self._get_word_embeddings()
    
    def _get_word_embeddings(self):
        """Return dummy word embeddings for all classes"""
        # In real implementation, this would load from a file
        emb_dim = 300  # Typical GloVe dimension
        return np.random.randn(self.num_classes, emb_dim).astype(np.float32)
    
    def _create_dummy_data(self):
        # Create dummy data for training
        num_dummy = 100
        self.imgs = [f"dummy_{i}.jpg" for i in range(num_dummy)]
        self.labels = np.random.randint(0, 2, size=(num_dummy, self.num_classes)).astype(np.float32)
        self.coco_ids = list(range(num_dummy))
    
    def _load_coco(self):
        if not os.path.exists(self.ann_file):
            print(f"Warning: Annotation file {self.ann_file} not found. Will use dummy data.")
            return
        
        if not os.path.exists(self.img_dir):
            print(f"Warning: Image directory {self.img_dir} not found. Will use dummy data.")
            return
            
        print(f"Loading COCO annotations from {self.ann_file}")
        try:
            with open(self.ann_file, 'r') as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"Error loading annotations: {e}")
            return
        
        # Create category id mapping (COCO has non-contiguous ids)
        categories = dataset['categories']
        self.coco_id_to_index = {cat['id']: i for i, cat in enumerate(categories)}
        
        # Create image dictionary for faster access
        img_dict = {img['id']: img for img in dataset['images']}
        
        # Group annotations by image
        img_to_anns = {}
        for ann in dataset['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Create dataset items
        valid_img_count = 0
        skipped_img_count = 0
        
        for img_id, anns in img_to_anns.items():
            if img_id not in img_dict:
                continue
                
            img_info = img_dict[img_id]
            file_name = img_info['file_name']
            img_path = os.path.join(self.img_dir, file_name)
            
            # Skip if image doesn't exist
            if not os.path.exists(img_path):
                skipped_img_count += 1
                if skipped_img_count < 10:  # Only print first few to avoid spam
                    print(f"Image not found: {img_path}")
                continue
                
            # Create multi-label vector for this image
            target = np.zeros(self.num_classes, dtype=np.float32)
            for ann in anns:
                category_id = ann['category_id']
                if category_id in self.coco_id_to_index:
                    label_idx = self.coco_id_to_index[category_id]
                    target[label_idx] = 1.0
            
            self.imgs.append(img_path)
            self.labels.append(target)
            self.coco_ids.append(img_id)
            valid_img_count += 1
            
        if valid_img_count == 0:
            print(f"No valid images found in {self.img_dir}. Checked annotation file: {self.ann_file}")
            print(f"Found {len(img_dict)} images in annotations and {len(img_to_anns)} images with annotations.")
            print(f"Skipped {skipped_img_count} images that were not found on disk.")
            
        if len(self.labels) > 0:
            self.labels = np.array(self.labels)
        
        print(f"Loaded {len(self.imgs)} images with {self.num_classes} categories for {self.split}")
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        target = self.labels[index]
        
        # For dummy data, generate a random image
        if img_path.startswith("dummy_"):
            img = Image.new('RGB', (224, 224), color=(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)))
        else:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Create a blank image as fallback
                img = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # Convert label to tensor
        target_tensor = torch.from_numpy(target)
        
        # The word_embeddings need to be the proper shape
        # It should have shape [batch=1, num_classes, embedding_dim]
        inp = torch.from_numpy(self.word_embeddings).float().unsqueeze(0)
        
        # Use empty tensor instead of None for the middle element to avoid collate errors
        empty_tensor = torch.zeros(1)
        
        # The GCNMultiLabelMAPEngine expects a triple:
        # - feature (img): Used for feature extraction in ResNet
        # - out (middle): Not used but must be a tensor
        # - input (inp): The word embeddings for GCN
        return (img, empty_tensor, inp), target_tensor
    
    def __len__(self):
        return len(self.imgs)

class TrainDataset(COCODataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        super().__init__(data_path, 'train', transform, target_transform)

class ValDataset(COCODataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        super().__init__(data_path, 'val', transform, target_transform)