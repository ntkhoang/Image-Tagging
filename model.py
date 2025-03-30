import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Parameter
import math

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BaseMLGCN(nn.Module):
    """
    Base ML-GCN model with ResNet-50 as feature extractor
    """
    def __init__(self, num_classes, in_channel=300, dropout=0.5):
        super(BaseMLGCN, self).__init__()
        # ResNet-50 for feature extraction
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # GCN layers for label correlation
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)  # 2048-D to match ResNet output
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        # Image normalization parameters (ImageNet)
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, image_features, label_embeddings, adj_matrix):
        # Extract image features
        img_features = self.features(image_features)
        img_features = self.pooling(img_features)
        img_features = img_features.view(img_features.size(0), -1)
        
        # GCN to process label correlations
        label_features = self.gc1(label_embeddings, adj_matrix)
        label_features = self.relu(label_features)
        label_features = self.dropout(label_features)
        label_features = self.gc2(label_features, adj_matrix)
        
        # Generate final predictions
        label_features = label_features.transpose(0, 1)
        predictions = torch.matmul(img_features, label_features)
        return predictions
    
    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
        ]


class EnhancedMLGCN(nn.Module):
    """
    Enhanced ML-GCN with Vision Transformer (ViT) as feature extractor
    """
    def __init__(self, num_classes, in_channel=300, dropout=0.5):
        super(EnhancedMLGCN, self).__init__()
        # Use ViT-Base as feature extractor
        self.vit = models.vit_b_16(pretrained=True)
        # Remove the classification head, keep only the feature extractor
        self.vit.heads = nn.Identity()  # ViT outputs 768-D features
        
        # GCN layers for label correlation
        self.gc1 = GraphConvolution(in_channel, 512)
        self.gc2 = GraphConvolution(512, 768)  # 768-D to match ViT output
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        # Image normalization parameters (ImageNet)
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, image_features, label_embeddings, adj_matrix):
        # Extract image features using ViT
        img_features = self.vit(image_features)  # B x 768
        
        # GCN to process label correlations
        label_features = self.gc1(label_embeddings, adj_matrix)
        label_features = self.relu(label_features)
        label_features = self.dropout(label_features)
        label_features = self.gc2(label_features, adj_matrix)
        
        # Generate final predictions
        label_features = label_features.transpose(0, 1)
        predictions = torch.matmul(img_features, label_features)
        return predictions
    
    def get_config_optim(self, lr, lrp):
        # Freeze early layers of ViT, update later ones
        return [
            {'params': self.vit.parameters(), 'lr': lr * lrp},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
        ]


# Loss functions
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


# Utility functions for adjacency matrix creation
def create_label_adjacency_matrix(label_vectors, threshold=0.4, p=0.2):
    """
    Create adjacency matrix based on label co-occurrence
    
    Args:
        label_vectors: Multi-hot encoded label vectors (samples x num_classes)
        threshold: Threshold for binarizing co-occurrence probabilities
        p: Weight for re-weighting
        
    Returns:
        Normalized adjacency matrix
    """
    num_classes = label_vectors.shape[1]
    
    # Co-occurrence counts
    co_occur = torch.zeros((num_classes, num_classes))
    class_count = torch.zeros(num_classes)
    
    for sample in label_vectors:
        indices = torch.where(sample == 1)[0]
        class_count[indices] += 1
        
        for i in indices:
            for j in indices:
                if i != j:
                    co_occur[i, j] += 1
    
    # Conditional probabilities
    prob = torch.zeros_like(co_occur)
    for i in range(num_classes):
        if class_count[i] > 0:
            prob[i] = co_occur[i] / class_count[i]
    
    # Binarize and re-weight
    adj = (prob >= threshold).float()
    adj_weighted = torch.zeros_like(adj)
    
    for i in range(num_classes):
        row_sum = adj[i].sum()
        if row_sum == 0:
            adj_weighted[i, i] = 1.0
        else:
            adj_weighted[i] = adj[i] * (p / row_sum)
            adj_weighted[i, i] = 1 - p
    
    # Normalize adjacency matrix
    D = torch.pow(adj_weighted.sum(1), -0.5)
    D = torch.diag(D)
    norm_adj = torch.matmul(torch.matmul(D, adj_weighted), D)
    
    return norm_adj


def enhance_adjacency_with_bert(adj_matrix, label_texts, bert_model):
    """
    Enhance adjacency matrix with BERT embeddings
    
    Args:
        adj_matrix: Original adjacency matrix based on co-occurrence
        label_texts: List of label text strings
        bert_model: Pre-trained BERT model for embeddings
        
    Returns:
        Enhanced adjacency matrix
    """
    # Get BERT embeddings for each label
    embeddings = []
    for text in label_texts:
        # This is a simplified version - in practice, use proper tokenizer
        inputs = bert_model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        embeddings.append(embedding)
    
    # Stack embeddings
    embeddings = torch.stack(embeddings)
    
    # Calculate cosine similarity
    norm = embeddings.norm(dim=1, keepdim=True)
    similarity = torch.mm(embeddings, embeddings.t()) / (torch.mm(norm, norm.t()) + 1e-8)
    
    # Combine with original adjacency matrix
    enhanced_adj = 0.7 * adj_matrix + 0.3 * similarity
    
    return enhanced_adj 