import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from tqdm import tqdm
import pickle

from models import gcn_resnet101
from engine import GCNMultiLabelMAPEngine
from util import gen_A, gen_adj, AveragePrecisionMeter, Warp, MultiScaleCrop
from dataset import TrainDataset, ValDataset

# Parse arguments
parser = argparse.ArgumentParser(description='ML-GCN Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--model-path', default='checkpoint', type=str, help='Path to save models')
parser.add_argument('--adj-file', default='data/adj_matrix.pkl', type=str, help='Path to the adjacency matrix pickle file')
parser.add_argument('--num-classes', default=80, type=int, help='Number of classes')
parser.add_argument('--word-dim', default=300, type=int, help='Dimension of word embeddings')
parser.add_argument('--batch-size', default=16, type=int, help='Batch size for training')
parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers')
parser.add_argument('--epochs', default=20, type=int, help='Number of total epochs to run')
parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
parser.add_argument('--lr-decay', default=0.1, type=float, help='Learning rate decay')
parser.add_argument('--epoch-decay', default=[30, 60], type=int, nargs='+', help='Epochs after which to decay learning rate')
parser.add_argument('--resume', default='', type=str, help='Path to latest checkpoint')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate model on validation set')
parser.add_argument('--print-freq', default=100, type=int, help='Print frequency')
parser.add_argument('--threshold', default=0.4, type=float, help='Threshold for adjacency matrix')
parser.add_argument('--weight', default=0.2, type=float, help='Weight for adjacency matrix')
parser.add_argument('--image-size', default=448, type=int, help='Input image size')

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

def load_word_embeddings(word_file, num_classes):
    """Load pre-trained word embeddings for class labels"""
    try:
        with open(word_file, 'rb') as f:
            word_embeddings = pickle.load(f)
        return word_embeddings
    except (FileNotFoundError, pickle.PickleError):
        # Fallback: return random embeddings
        print(f"Warning: Could not load word embeddings from {word_file}. Using random embeddings.")
        return np.random.randn(num_classes, 300)  # 300 is typical for GloVe

def main():
    args = parser.parse_args()
    
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Load adjacency matrix
    if os.path.exists(args.adj_file):
        adj_matrix = gen_A(args.num_classes, args.threshold, args.adj_file)
    else:
        print(f"Warning: Adjacency file {args.adj_file} not found. Creating an identity matrix.")
        adj_matrix = np.identity(args.num_classes)
    
    # Load word embeddings for classes
    word_embeddings = load_word_embeddings('data/word_embeddings.pkl', args.num_classes)
    
    # Create model
    model = gcn_resnet101(num_classes=args.num_classes, t=args.threshold, adj_file=args.adj_file, 
                          in_channel=args.word_dim, pretrained=True)
    
    # Define loss function, optimizer, and learning rate scheduler
    criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=0)
    
    # Initialize the GCN Engine
    state = {
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'max_epochs': args.epochs,
        'evaluate': args.evaluate,
        'resume': args.resume,
        'workers': args.workers,
        'print_freq': args.print_freq,
        'epoch_step': args.epoch_decay,
        'save_model_path': args.model_path
    }
    engine = GCNMultiLabelMAPEngine(state)
    
    # Get optimizer with different learning rates for different parts of the model
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, 0.1), lr=args.lr, 
                               momentum=0.9, weight_decay=1e-4)
    
    # Load datasets
    if not args.evaluate:
        train_dataset = TrainDataset(args.data)
        val_dataset = ValDataset(args.data)
        
        # Run training
        engine.learning(model, criterion, train_dataset, val_dataset, optimizer)
    else:
        val_dataset = ValDataset(args.data)
        
        # Run evaluation
        engine.validate(val_dataset, model, criterion)

# Example of how to use the model
def example_usage():
    # Hyperparameters
    NUM_LABELS = 80
    EMBEDDING_DIM = 300
    
    # Load label embeddings (e.g., GloVe)
    label_embeddings = np.random.randn(NUM_LABELS, EMBEDDING_DIM)  # Replace with real data
    
    # Create model
    model = gcn_resnet101(NUM_LABELS, t=0.4, pretrained=True)
    model.eval()
    
    # Example forward pass
    dummy_input = torch.randn(1, 3, 448, 448)
    with torch.no_grad():
        # In real usage, you would pass features and word embeddings
        features = model.features(dummy_input)
        pooled_features = model.pooling(features).view(1, -1)
        
        # Assuming we have word embeddings and adjacency matrix
        dummy_word_embeddings = torch.FloatTensor(label_embeddings).unsqueeze(0)
        dummy_adj = gen_adj(torch.eye(NUM_LABELS))
        
        # Final prediction
        outputs = model(dummy_input, [dummy_word_embeddings])
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        
        print(f"Example output shape: {outputs.shape}")
        print(f"Example predictions: {predicted.sum().item()} labels detected")

if __name__ == "__main__":
    main()
    # Uncomment to run example usage
    # example_usage()