import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Import GCN components from the gcn folder
from gcn.model import TwoLayerGCN

class SimplifiedGCN(nn.Module):
    """
    A simplified GCN implementation that works with batched image features
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(SimplifiedGCN, self).__init__()
        
        # Regular feed-forward layers for image features
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Apply dropout to input
        x = self.dropout(x)
        
        # First layer
        x = self.fc1(x)
        x = self.relu(x)
        
        # Second layer
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ImageGCNSimple(nn.Module):
    """
    A simplified architecture combining image feature extraction with GCN for label correlation
    without PyTorch Geometric dependency
    """
    def __init__(self, num_classes, feature_dim=2048, hidden_dim=1024, dropout=0.5, 
                 use_vit=False, device='cuda', train_backbone=False, img_size=224):
        super(ImageGCNSimple, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.train_backbone = train_backbone
        self.img_size = img_size

        # Image feature extractor
        if use_vit:
            # Use Vision Transformer
            if img_size == 224:
                # Standard ViT size
                vit = models.vit_b_16(weights='DEFAULT')
                vit.heads = nn.Identity()  # Remove classification head
            else:
                # Custom size ViT with position embedding interpolation
                print(f"Creating custom ViT with image size: {img_size}x{img_size}")
                
                # First, we get the pretrained model
                vit = models.vit_b_16(weights='DEFAULT')
                
                # We need to modify the model to accept a different input size
                # 1. Change the image size in the model
                vit.image_size = img_size
                
                # 2. Modify the patch embedding to handle the new image size
                # The number of patches will be different with the new image size
                # but the patch size remains the same (16x16)
                
                # Calculate the new sequence length
                patch_size = vit.patch_size
                num_patches = (img_size // patch_size) ** 2
                
                # 3. Update positional embeddings for the encoder
                # Get the current positional embedding
                pos_embed = vit.encoder.pos_embedding
                
                # Separate class token and patch tokens
                class_pos_embed = pos_embed[:, 0:1]  # Class token positional embedding
                patch_pos_embed = pos_embed[:, 1:]   # Patch tokens positional embeddings
                
                # Compute the number of patches in the original model
                num_patches_orig = patch_pos_embed.shape[1]
                orig_size = int(num_patches_orig ** 0.5)
                
                if num_patches != num_patches_orig:
                    # Resize the positional embeddings to match the new number of patches
                    # Reshape to a grid
                    patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, -1)
                    
                    # Interpolate to the new grid size
                    new_size = img_size // patch_size
                    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # [1, dim, h, w]
                    patch_pos_embed = nn.functional.interpolate(
                        patch_pos_embed, 
                        size=(new_size, new_size), 
                        mode='bicubic', 
                        align_corners=False
                    )
                    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)  # [1, h, w, dim]
                    patch_pos_embed = patch_pos_embed.reshape(1, num_patches, -1)
                    
                    # Create new positional embedding
                    new_pos_embed = torch.cat([class_pos_embed, patch_pos_embed], dim=1)
                    
                    # Update the positional embedding
                    vit.encoder.pos_embedding = nn.Parameter(new_pos_embed)
                
                # Now we need to patch the _process_input method in the ViT model to bypass the size check
                original_process_input = vit._process_input
                
                def new_process_input(self, x):
                    # Skip the size check but keep the rest of the function
                    n, c, h, w = x.shape
                    p = self.patch_size
                    
                    # We just ensure divisibility and reshape
                    torch._assert(h % p == 0, f"Input image height {h} is not divisible by patch size {p}!")
                    torch._assert(w % p == 0, f"Input image width {w} is not divisible by patch size {p}!")
                    
                    # Actually now using the current image size instead of the original 224
                    #torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
                    #torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
                    
                    # Extract patches
                    x = self.conv_proj(x)  # Shape: [batch_size, hidden_dim, grid_size, grid_size]
                    x = x.reshape(n, self.hidden_dim, -1).permute(0, 2, 1)  # Shape: [batch_size, num_patches, hidden_dim]
                    
                    # Add class token
                    x = torch.cat([self.class_token.expand(n, -1, -1), x], dim=1)
                    
                    return x
                
                # Replace the method
                import types
                vit._process_input = types.MethodType(new_process_input, vit)
                
                # Replace the head for feature extraction
                vit.heads = nn.Identity()
                
                print("Successfully created custom ViT with size", img_size)
                
            self.feature_extractor = vit
            self.feature_dim = 768
        else:
            # Use ResNet-50
            resnet = models.resnet50(weights='DEFAULT')
            modules = list(resnet.children())[:-1]  # Remove FC layer
            self.feature_extractor = nn.Sequential(*modules)
            self.feature_dim = 2048
        
        # Set trainability of the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = train_backbone
            
        # Use our simplified GCN implementation instead of TwoLayerGCN
        self.gcn = SimplifiedGCN(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            output_size=num_classes,
            dropout=dropout
        )
        
        # Linear layer to project image features to graph input
        self.projection = nn.Linear(self.feature_dim, feature_dim)
        
        # Normalization stats for images
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def extract_features(self, images):
        """
        Extract image features from the feature extractor.
        If the backbone is frozen (train_backbone is False),
        use torch.no_grad() to save memory.
        """
        if self.train_backbone:
            # Backbone is trainable, so compute gradients normally.
            if isinstance(self.feature_extractor, nn.Sequential):
                features = self.feature_extractor(images)
                features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
            else:
                features = self.feature_extractor(images)  # For ViT
        else:
            # Backbone is frozen, disable gradient computation.
            with torch.no_grad():
                if isinstance(self.feature_extractor, nn.Sequential):
                    features = self.feature_extractor(images)
                    features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
                else:
                    features = self.feature_extractor(images)  # For ViT
        return features
    
    def forward(self, images, adj_matrix=None):
        """
        Forward pass of the model
        
        Args:
            images: Input images (batch_size, 3, H, W)
            adj_matrix: Adjacency matrix (not used in this simplified implementation)
            
        Returns:
            Class predictions
        """
        # Extract image features
        batch_size = images.size(0)
        image_features = self.extract_features(images)  # [batch_size, feature_dim]
        
        # Project to input dimension if needed
        image_features = self.projection(image_features)  # [batch_size, feature_dim]
        
        # Ensure all tensors are float32
        image_features = image_features.to(dtype=torch.float32)
        
        # Process through simplified GCN
        outputs = self.gcn(image_features)
        
        return outputs

class ImageTaggingDataset(Dataset):
    """
    Dataset for image tagging with GCN
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

class ImageGCNTrainer:
    """
    Trainer for ImageGCN model
    """
    def __init__(self, model, device='cuda', learning_rate=0.001, weight_decay=5e-4):
        self.model = model
        self.device = device
        self.model.to(device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train_epoch(self, train_loader, adj_matrix=None):
        """
        Train the model for a single epoch
        
        Args:
            train_loader: DataLoader for training data
            adj_matrix: Adjacency matrix for GCN (not used in simplified implementation)
            
        Returns:
            Average training loss for the epoch
        """
        # Set model to training mode
        self.model.train()
        
        train_loss = 0.0
        
        # Train on batches
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass (no need for adjacency matrix)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
        
        # Return average loss
        avg_loss = train_loss / len(train_loader)
        return avg_loss
    
    def train(self, train_loader, val_loader, adj_matrix=None, num_epochs=100):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            adj_matrix: Adjacency matrix for GCN (not used in simplified implementation)
            num_epochs: Number of epochs to train
        
        Returns:
            Best validation accuracy
        """
        # Training loop
        best_val_accuracy = 0.0
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    
                    # Calculate accuracy (for multi-label)
                    preds = torch.sigmoid(outputs) > 0.5
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.numel()
            
            # Print epoch results
            val_accuracy = val_correct / val_total
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), "best_model.pt")
                print(f"  Saved new best model with accuracy: {best_val_accuracy:.4f}")
        
        return best_val_accuracy
    
    def evaluate(self, test_loader, adj_matrix=None):
        """
        Evaluate the model on test data
        
        Args:
            test_loader: DataLoader for test data
            adj_matrix: Adjacency matrix for GCN (not used in simplified implementation)
            
        Returns:
            Test accuracy, precision, recall, F1
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass (no need for adjacency matrix)
                outputs = self.model(images)
                preds = torch.sigmoid(outputs) > 0.5
                
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        # Calculate metrics
        correct = (all_preds == all_labels).sum().item()
        total = all_labels.numel()
        accuracy = correct / total
        
        # Calculate per-class metrics
        true_positives = (all_preds & all_labels.bool()).sum(dim=0)
        pred_positives = all_preds.sum(dim=0)
        actual_positives = all_labels.sum(dim=0)
        
        # Avoid division by zero
        precision = true_positives.float() / torch.clamp(pred_positives.float(), min=1e-10)
        recall = true_positives.float() / torch.clamp(actual_positives.float(), min=1e-10)
        f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-10)
        
        # Average metrics
        avg_precision = precision.mean().item()
        avg_recall = recall.mean().item()
        avg_f1 = f1.mean().item()
        
        return {
            'accuracy': accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'per_class_precision': precision.numpy(),
            'per_class_recall': recall.numpy(),
            'per_class_f1': f1.numpy()
        }

def create_adjacency_matrix(labels, threshold=0.5):
    """
    Create adjacency matrix based on label co-occurrence
    
    Args:
        labels: Binary label matrix (num_samples x num_classes)
        threshold: Threshold for co-occurrence probability
        
    Returns:
        Normalized adjacency matrix
    """
    num_classes = labels.shape[1]
    
    # Count co-occurrences
    co_occur = np.zeros((num_classes, num_classes))
    class_count = np.zeros(num_classes)
    
    for sample in labels:
        # Get indices of positive labels
        pos_indices = np.where(sample == 1)[0]
        class_count[pos_indices] += 1
        
        # Count co-occurrences
        for i in pos_indices:
            for j in pos_indices:
                if i != j:
                    co_occur[i, j] += 1
    
    # Calculate co-occurrence probabilities
    prob = np.zeros_like(co_occur)
    for i in range(num_classes):
        if class_count[i] > 0:
            prob[i] = co_occur[i] / class_count[i]
    
    # Apply threshold
    adj = (prob >= threshold).astype(np.float32)
    
    # Add self-connections
    adj = adj + np.eye(num_classes)
    
    # Normalize adjacency matrix (D^{-1/2} A D^{-1/2})
    rowsum = adj.sum(axis=1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    return torch.FloatTensor(normalized_adj)

def main():
    # Example usage
    print("ImageGCN Simplified Implementation")
    print("This version avoids using PyTorch Geometric")
    print("Example setup:")
    print("1. Load your dataset")
    print("2. Create adjacency matrix from label co-occurrences")
    print("3. Initialize ImageGCNSimple model")
    print("4. Train and evaluate the model")

if __name__ == "__main__":
    main() 