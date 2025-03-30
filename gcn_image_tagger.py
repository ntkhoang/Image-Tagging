import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch_geometric.data import Data, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# Import GCN components from the gcn folder
from gcn.model import GCN, TwoLayerGCN
from gcn.trainer import RunConfig

class ImageGCN(nn.Module):
    """
    An integrated architecture combining image feature extraction with GCN for label correlation
    """
    def __init__(self, num_classes, feature_dim=2048, hidden_dim=1024, dropout=0.5, 
                 use_vit=False, device='cuda'):
        super(ImageGCN, self).__init__()
        self.device = device
        self.num_classes = num_classes
        
        # Image feature extractor
        if use_vit:
            # Use Vision Transformer
            vit = models.vit_b_16(pretrained=True)
            vit.heads = nn.Identity()  # Remove classification head
            self.feature_extractor = vit
            self.feature_dim = 768
        else:
            # Use ResNet-50
            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]  # Remove FC layer
            self.feature_extractor = nn.Sequential(*modules)
            self.feature_dim = 2048
        
        # GCN for label correlation
        self.gcn = TwoLayerGCN(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            output_size=num_classes,
            dropout=dropout
        )
        
        # Normalization stats for images
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def extract_features(self, images):
        """
        Extract image features from the feature extractor
        """
        with torch.no_grad():
            if isinstance(self.feature_extractor, nn.Sequential):
                features = self.feature_extractor(images)
                features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
            else:
                features = self.feature_extractor(images)  # For ViT
        return features
    
    def forward(self, images, adj_matrix):
        """
        Forward pass of the model
        
        Args:
            images: Input images (batch_size, 3, H, W)
            adj_matrix: Adjacency matrix for GCN
            
        Returns:
            Class predictions
        """
        # Extract image features
        image_features = self.extract_features(images)
        
        # Convert adjacency matrix to proper format for GCN
        # Handle the case where edge_index is used directly in gcn/model.py
        # Instead of passing edge_index, pass normalized adjacency matrix
        # and convert all tensors to float type
        
        # Convert to format expected by our GCN implementation
        adjacency_hat = adj_matrix.to(dtype=torch.float32)
        features = image_features.to(dtype=torch.float32)
        
        # Call GCN directly with the correct types
        outputs = self.gcn(features, adjacency_hat)
        
        return outputs

def adj_matrix_to_edge_index(adj_matrix):
    """
    Convert adjacency matrix to edge index format used by PyTorch Geometric
    
    Args:
        adj_matrix: Adjacency matrix (N x N)
        
    Returns:
        edge_index: Edge index tensor (2 x E)
    """
    # Get edges where adjacency is non-zero
    edge_index = torch.nonzero(adj_matrix > 0, as_tuple=True)
    edge_index = torch.stack(edge_index)
    
    return edge_index

class ImageTaggingDataset(torch.utils.data.Dataset):
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
        image = Image.open(img_path).convert('RGB')
        
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
    
    def train(self, train_loader, val_loader, adj_matrix, num_epochs=100):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            adj_matrix: Adjacency matrix for GCN
            num_epochs: Number of epochs to train
        
        Returns:
            Best validation accuracy
        """
        # Prepare optimizer and loss function
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        criterion = nn.BCEWithLogitsLoss()
        
        # Move adjacency matrix to device
        adj_matrix = adj_matrix.to(self.device)
        
        # Training loop
        best_val_accuracy = 0.0
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images, adj_matrix)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy (for multi-label)
                preds = torch.sigmoid(outputs) > 0.5
                train_correct += (preds == labels).sum().item()
                train_total += labels.numel()
            
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
                    outputs = self.model(images, adj_matrix)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    
                    # Calculate accuracy (for multi-label)
                    preds = torch.sigmoid(outputs) > 0.5
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.numel()
            
            # Print epoch results
            train_accuracy = train_correct / train_total
            val_accuracy = val_correct / val_total
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), "best_model.pt")
                print(f"  Saved new best model with accuracy: {best_val_accuracy:.4f}")
        
        return best_val_accuracy
    
    def evaluate(self, test_loader, adj_matrix):
        """
        Evaluate the model on test data
        
        Args:
            test_loader: DataLoader for test data
            adj_matrix: Adjacency matrix for GCN
            
        Returns:
            Test accuracy, precision, recall, F1
        """
        self.model.eval()
        
        # Move adjacency matrix to device
        adj_matrix = adj_matrix.to(self.device)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images, adj_matrix)
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
        true_positives = (all_preds & all_labels).sum(dim=0)
        pred_positives = all_preds.sum(dim=0)
        actual_positives = all_labels.sum(dim=0)
        
        precision = true_positives / pred_positives
        recall = true_positives / actual_positives
        f1 = 2 * precision * recall / (precision + recall)
        
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
    
    # Normalize adjacency matrix
    rowsum = adj.sum(axis=1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    return torch.FloatTensor(normalized_adj)

def main():
    # Example usage
    print("ImageGCN model implementation. Import and use as needed.")
    print("Example setup:")
    print("1. Load your dataset")
    print("2. Create adjacency matrix from label co-occurrences")
    print("3. Initialize ImageGCN model")
    print("4. Train and evaluate the model")

if __name__ == "__main__":
    main() 