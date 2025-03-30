import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms


class MLGCNTrainer:
    """
    Trainer for ML-GCN models with support for pre-training and fine-tuning
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.scaler = GradScaler()  # For mixed precision training
    
    def pretraining(self, train_loader, val_loader, epochs=10, lr=0.001, 
                  temperature=0.07, log_interval=10, checkpoint_dir='checkpoints'):
        """
        Pre-train the model using contrastive learning
        
        Args:
            train_loader: DataLoader with image-text pairs
            val_loader: DataLoader for validation
            epochs: Number of training epochs
            lr: Learning rate
            temperature: Temperature parameter for contrastive loss
            log_interval: How often to log progress
            checkpoint_dir: Directory to save checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Define optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_idx, (images, texts) in enumerate(tqdm(train_loader)):
                images = images.to(self.device)
                texts = texts.to(self.device)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    # Extract image and text features
                    image_features = self.model.extract_image_features(images)
                    text_features = self.model.extract_text_features(texts)
                    
                    # Normalize features
                    image_features = nn.functional.normalize(image_features, dim=1)
                    text_features = nn.functional.normalize(text_features, dim=1)
                    
                    # Compute similarity matrix
                    logits = torch.matmul(image_features, text_features.t()) / temperature
                    
                    # Contrastive loss
                    labels = torch.arange(images.size(0)).to(self.device)
                    loss_i2t = nn.CrossEntropyLoss()(logits, labels)
                    loss_t2i = nn.CrossEntropyLoss()(logits.t(), labels)
                    loss = (loss_i2t + loss_t2i) / 2
                
                # Backward and optimize with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                train_loss += loss.item()
                
                if (batch_idx + 1) % log_interval == 0:
                    print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            print(f"Training Loss: {avg_train_loss:.4f}")
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for images, texts in tqdm(val_loader):
                    images = images.to(self.device)
                    texts = texts.to(self.device)
                    
                    # Extract features
                    image_features = self.model.extract_image_features(images)
                    text_features = self.model.extract_text_features(texts)
                    
                    # Normalize features
                    image_features = nn.functional.normalize(image_features, dim=1)
                    text_features = nn.functional.normalize(text_features, dim=1)
                    
                    # Compute similarity matrix
                    logits = torch.matmul(image_features, text_features.t()) / temperature
                    
                    # Contrastive loss
                    labels = torch.arange(images.size(0)).to(self.device)
                    loss_i2t = nn.CrossEntropyLoss()(logits, labels)
                    loss_t2i = nn.CrossEntropyLoss()(logits.t(), labels)
                    loss = (loss_i2t + loss_t2i) / 2
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Save checkpoint if validation loss improved
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, os.path.join(checkpoint_dir, 'pretraining_best_model.pth'))
                print(f"Saved new best model with loss {best_loss:.4f}")
    
    def finetune(self, train_loader, val_loader, adj_matrix, label_embeddings, 
                epochs=20, lr=0.0001, log_interval=10, checkpoint_dir='checkpoints'):
        """
        Fine-tune the model on a specific dataset using Asymmetric Loss
        
        Args:
            train_loader: DataLoader with images and labels
            val_loader: DataLoader for validation
            adj_matrix: Adjacency matrix for GCN
            label_embeddings: Label embeddings for GCN
            epochs: Number of training epochs
            lr: Learning rate
            log_interval: How often to log progress
            checkpoint_dir: Directory to save checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Move data to device
        adj_matrix = adj_matrix.to(self.device)
        label_embeddings = label_embeddings.to(self.device)
        
        # Define optimizer with different learning rates for different parts
        optimizer = optim.Adam(self.model.get_config_optim(lr, 0.1), lr=lr)
        
        # Define loss function
        criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=0)
        
        # Training loop
        best_map = 0.0
        
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
                # Unpack the data tuple (img, empty_tensor, inp)
                features = data[0].to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = self.model(features, label_embeddings, adj_matrix)
                    loss = criterion(outputs, targets)
                
                # Backward and optimize with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                self.scaler.step(optimizer)
                self.scaler.update()
                
                train_loss += loss.item()
                
                if (batch_idx + 1) % log_interval == 0:
                    print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            print(f"Training Loss: {avg_train_loss:.4f}")
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            all_outputs = []
            all_targets = []
            
            with torch.no_grad():
                for data, targets in tqdm(val_loader):
                    # Unpack the data tuple
                    features = data[0].to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(features, label_embeddings, adj_matrix)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    all_outputs.append(outputs.cpu())
                    all_targets.append(targets.cpu())
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Calculate mAP
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            ap = self.calculate_ap(all_outputs, all_targets)
            mean_ap = ap.mean().item()
            print(f"Validation mAP: {mean_ap:.4f}")
            
            # Save checkpoint if mAP improved
            if mean_ap > best_map:
                best_map = mean_ap
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'map': best_map,
                }, os.path.join(checkpoint_dir, 'finetuning_best_model.pth'))
                print(f"Saved new best model with mAP {best_map:.4f}")
    
    def evaluate(self, test_loader, adj_matrix, label_embeddings):
        """
        Evaluate the model on test data
        
        Args:
            test_loader: DataLoader with test images and labels
            adj_matrix: Adjacency matrix for GCN
            label_embeddings: Label embeddings for GCN
            
        Returns:
            mAP score and per-class AP
        """
        self.model.eval()
        all_outputs = []
        all_targets = []
        
        # Move data to device
        adj_matrix = adj_matrix.to(self.device)
        label_embeddings = label_embeddings.to(self.device)
        
        with torch.no_grad():
            for data, targets in tqdm(test_loader):
                # Unpack the data tuple
                features = data[0].to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(features, label_embeddings, adj_matrix)
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate Average Precision for each class
        ap = self.calculate_ap(all_outputs, all_targets)
        mean_ap = ap.mean().item()
        
        # Also calculate precision, recall, F1
        probs = torch.sigmoid(all_outputs)
        preds = (probs > 0.5).float()
        
        # Overall metrics
        TP = (preds * all_targets).sum()
        precision = TP / preds.sum() if preds.sum() > 0 else 0
        recall = TP / all_targets.sum() if all_targets.sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Per-class metrics
        class_precision = (preds * all_targets).sum(0) / preds.sum(0).clamp(min=1e-10)
        class_recall = (preds * all_targets).sum(0) / all_targets.sum(0).clamp(min=1e-10)
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall).clamp(min=1e-10)
        
        results = {
            'mAP': mean_ap,
            'AP': ap.numpy(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'class_precision': class_precision.numpy(),
            'class_recall': class_recall.numpy(),
            'class_f1': class_f1.numpy()
        }
        
        return results
    
    @staticmethod
    def calculate_ap(outputs, targets):
        """
        Calculate Average Precision for multi-label classification
        
        Args:
            outputs: Model predictions (N x C)
            targets: Ground truth labels (N x C)
            
        Returns:
            AP for each class
        """
        probs = torch.sigmoid(outputs)
        
        # Sort predictions
        sorted_probs, indices = torch.sort(probs, dim=0, descending=True)
        sorted_targets = torch.gather(targets, 0, indices)
        
        # Compute precision at each cutoff
        tp = torch.cumsum(sorted_targets, dim=0)
        fp = torch.cumsum(1 - sorted_targets, dim=0)
        
        precision = tp / (tp + fp)
        recall = tp / (targets.sum(0).unsqueeze(0) + 1e-10)
        
        # Compute AP using trapezoidal rule
        zero = torch.zeros(1, outputs.size(1)).to(outputs.device)
        recall_all = torch.cat([zero, recall, torch.ones(1, outputs.size(1)).to(outputs.device)], dim=0)
        precision_all = torch.cat([torch.zeros(1, outputs.size(1)).to(outputs.device), precision, zero], dim=0)
        
        # Compute area under PR curve
        ap = torch.zeros(outputs.size(1))
        for i in range(outputs.size(1)):
            for j in range(recall_all.size(0) - 1):
                ap[i] += (recall_all[j+1, i] - recall_all[j, i]) * precision_all[j, i]
        
        return ap


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) as described in the paper
    """
    def __init__(self, gamma_neg=2, gamma_pos=0, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.eps = eps
    
    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        pos_loss = targets * (1 - probabilities).pow(self.gamma_pos) * torch.log(probabilities + self.eps)
        neg_loss = (1 - targets) * probabilities.pow(self.gamma_neg) * torch.log(1 - probabilities + self.eps)
        return -torch.mean(pos_loss + neg_loss)


def get_transforms(image_size=224):
    """
    Get transforms for training and validation/testing
    
    Args:
        image_size: Size to resize images to
        
    Returns:
        train_transform, val_transform
    """
    # ImageNet normalization values
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation/test transforms
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, val_transform 