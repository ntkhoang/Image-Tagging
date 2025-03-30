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
import torch.utils.data as data
from PIL import Image
import json
from transformers import BertModel, BertTokenizer

from models import gcn_resnet101
from engine import GCNMultiLabelMAPEngine
from util import gen_A, gen_adj, AveragePrecisionMeter, Warp, MultiScaleCrop
from dataset import TrainDataset, ValDataset, COCODataset
from model import BaseMLGCN, EnhancedMLGCN, create_label_adjacency_matrix, enhance_adjacency_with_bert
from trainer import MLGCNTrainer, get_transforms, AsymmetricLoss

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

def parse_args():
    parser = argparse.ArgumentParser(description='ML-GCN for Image Tagging with Transformers')
    
    # Dataset parameters
    parser.add_argument('--data-path', type=str, default='coco', help='Path to dataset')
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'openimages'], help='Dataset to use')
    parser.add_argument('--num-classes', type=int, default=80, help='Number of classes')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='base', choices=['base', 'enhanced'], 
                        help='Model architecture to use (base: ResNet50+GCN, enhanced: ViT+GCN)')
    parser.add_argument('--adj-method', type=str, default='cooccur', choices=['cooccur', 'bert'],
                        help='Method to create adjacency matrix (cooccur: co-occurrence, bert: BERT enhanced)')
    parser.add_argument('--embedding-dim', type=int, default=300, help='Dimension of label embeddings')
    
    # Training parameters
    parser.add_argument('--pretrain', action='store_true', help='Perform pre-training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda, cpu)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Evaluation parameters
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--test-only', action='store_true', help='Perform testing only')
    
    return parser.parse_args()

def get_label_names(args, dataset_name):
    """
    Get label names for the dataset
    """
    if dataset_name == 'coco':
        # COCO has 80 classes
        coco_file = os.path.join(args.data_path, "annotations", "instances_train2017.json")
        if os.path.exists(coco_file):
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            categories = coco_data['categories']
            label_names = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
        else:
            # Fallback to standard COCO classes
            label_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
                          "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
                          "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
                          "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                          "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
                          "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
                          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
                          "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                          "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
                          "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    else:
        # For OpenImages, load the label names from a file or define them
        # This is a placeholder - in practice, you would load the actual labels
        label_names = [f"label_{i}" for i in range(args.num_classes)]
    
    return label_names

def load_or_create_adjacency_matrix(args, train_dataset):
    """
    Load or create adjacency matrix for GCN
    """
    adj_file = f"{args.checkpoint_dir}/adj_matrix_{args.dataset}_{args.adj_method}.pt"
    
    if os.path.exists(adj_file):
        print(f"Loading adjacency matrix from {adj_file}")
        adj_matrix = torch.load(adj_file)
    else:
        print("Creating adjacency matrix...")
        # Collect all labels for co-occurrence calculation
        all_labels = []
        loader = data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        for _, targets in tqdm(loader):
            all_labels.append(targets)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Create adjacency matrix
        adj_matrix = create_label_adjacency_matrix(all_labels)
        
        # Enhance with BERT if specified
        if args.adj_method == 'bert':
            print("Enhancing adjacency matrix with BERT embeddings...")
            label_names = get_label_names(args, args.dataset)
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model.tokenizer = tokenizer
            adj_matrix = enhance_adjacency_with_bert(adj_matrix, label_names, bert_model)
        
        # Save adjacency matrix
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(adj_matrix, adj_file)
    
    return adj_matrix

def load_or_create_label_embeddings(args):
    """
    Load or create label embeddings for GCN
    """
    emb_file = f"{args.checkpoint_dir}/label_embeddings_{args.dataset}.pt"
    
    if os.path.exists(emb_file):
        print(f"Loading label embeddings from {emb_file}")
        label_embeddings = torch.load(emb_file)
    else:
        print("Creating label embeddings...")
        # For simplicity, we'll use BERT embeddings for the labels
        label_names = get_label_names(args, args.dataset)
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        embeddings = []
        for name in tqdm(label_names):
            inputs = tokenizer(name, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            embeddings.append(embedding)
        
        label_embeddings = torch.stack(embeddings)
        
        # Save label embeddings
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(label_embeddings, emb_file)
    
    return label_embeddings

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create transforms
    train_transform, val_transform = get_transforms(image_size=224)
    
    # Create datasets
    train_dataset = TrainDataset(args.data_path, transform=train_transform)
    val_dataset = ValDataset(args.data_path, transform=val_transform)
    
    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    # Load or create adjacency matrix
    adj_matrix = load_or_create_adjacency_matrix(args, train_dataset)
    
    # Load or create label embeddings
    label_embeddings = load_or_create_label_embeddings(args)
    
    # Create model
    if args.model == 'base':
        model = BaseMLGCN(args.num_classes, in_channel=768)
    else:
        model = EnhancedMLGCN(args.num_classes, in_channel=768)
    
    # Create trainer
    trainer = MLGCNTrainer(model, device=device)
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Train or evaluate
    if not args.test_only:
        if args.pretrain:
            # This is just a placeholder - you would need your own pretraining data
            print("Pretraining with contrastive learning...")
            # In practice, you would have a dataloader with image-text pairs
            # trainer.pretraining(pretraining_loader, val_loader, epochs=10, lr=0.001)
            print("Pretraining not implemented in this example")
        
        # Fine-tune
        print("Fine-tuning the model...")
        trainer.finetune(
            train_loader, val_loader, adj_matrix, label_embeddings,
            epochs=args.epochs, lr=args.lr, checkpoint_dir=args.checkpoint_dir
        )
    
    # Evaluate
    if args.evaluate or args.test_only:
        print("Evaluating the model...")
        results = trainer.evaluate(val_loader, adj_matrix, label_embeddings)
        print(f"mAP: {results['mAP']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1: {results['f1']:.4f}")

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