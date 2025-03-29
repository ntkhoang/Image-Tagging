import torch
import torch.nn as nn
import torchvision
import numpy as np

# 1. Define GCN Layer
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
    
    def forward(self, H, A_hat):
        H_transformed = self.linear(H)
        H_next = torch.matmul(A_hat, H_transformed)
        return self.activation(H_next)

# 2. Define GCN Module
class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.gcn1 = GCNLayer(in_features, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, out_features)
    
    def forward(self, Z, A_hat):
        H = self.gcn1(Z, A_hat)
        H = self.gcn2(H, A_hat)
        return H

# 3. Define ML-GCN Model
class MLGCN(nn.Module):
    def __init__(self, num_labels, embedding_dim, adj_matrix, label_embeddings, resnet_out_dim=2048, gcn_hidden=1024):
        super().__init__()
        # Image feature extractor (ResNet-50 up to layer4)
        self.resnet = nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[:-2])
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        # GCN for classifier generation
        self.gcn = GCN(embedding_dim, gcn_hidden, resnet_out_dim)
        # Register buffers for fixed parameters
        self.register_buffer('adj_matrix', torch.FloatTensor(adj_matrix))
        self.register_buffer('label_embeddings', torch.FloatTensor(label_embeddings))
    
    def forward(self, x):
        # Image features (B, 2048)
        x = self.resnet(x)
        x = self.global_pool(x).flatten(1)
        # Generate classifiers via GCN (C, 2048)
        W = self.gcn(self.label_embeddings, self.adj_matrix)
        # Calculate logits (B, C)
        logits = torch.matmul(x, W.T)
        return logits

# 4. Asymmetric Loss (ASL)
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=2, gamma_pos=0, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.eps = eps
    
    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        pos_loss = targets * (1 - probabilities).pow(self.gamma_pos) * torch.log(probabilities + self.eps)
        neg_loss = (1 - targets) * probabilities.pow(self.gamma_neg) * torch.log(1 - probabilities + self.eps)
        return -torch.mean(pos_loss + neg_loss)

# 5. Precompute Adjacency Matrix (Example)
def build_adjacency_matrix(labels, num_classes, tau=0.4, p=0.2):
    # labels: List[List[int]] indicating label indices per sample
    M = np.zeros((num_classes, num_classes))
    N = np.zeros(num_classes)
    for sample in labels:
        for i in sample:
            N[i] += 1
            for j in sample:
                if i != j:
                    M[i, j] += 1
    # Compute conditional probabilities
    P = np.divide(M, N[:, None], out=np.zeros_like(M), where=N[:, None]!=0)
    # Binarize and re-weight
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

# Example Usage
if __name__ == "__main__":
    # Hyperparameters
    NUM_LABELS = 1000  # Example
    EMBEDDING_DIM = 300
    
    # Load label embeddings (e.g., GloVe)
    label_embeddings = np.random.randn(NUM_LABELS, EMBEDDING_DIM)  # Replace with real data
    
    # Simulate training labels to build adjacency matrix
    train_labels = [np.random.randint(0, NUM_LABELS, size=5) for _ in range(1000)]  # Replace with real data
    adj_matrix = build_adjacency_matrix(train_labels, NUM_LABELS)
    
    # Initialize model
    model = MLGCN(NUM_LABELS, EMBEDDING_DIM, adj_matrix, label_embeddings)
    criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Freeze ResNet layers except layer4
    for name, param in model.resnet.named_parameters():
        if 'layer4' not in name:
            param.requires_grad = False
    
    # Training loop example
    for epoch in range(20):
        for images, targets in train_dataloader:  # Replace with actual dataloader
            logits = model(images)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()