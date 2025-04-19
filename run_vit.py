import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr
import re
from collections import OrderedDict

from gcn_image_tagger_simplified import ImageGCNSimple

# Set the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Will be initialized later based on detected image size
transform = None

def detect_model_specs(model_path):
    """
    Detect model specifications from the state dictionary
    
    Args:
        model_path: Path to saved model weights
        
    Returns:
        img_size: Detected image size
        num_classes: Detected number of classes
    """
    print(f"Detecting model specifications from {model_path}")
    try:
        # Load state dict to analyze its shape
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        
        # Check if this is a checkpoint file with multiple dictionaries
        if 'model_state_dict' in state_dict:
            print("Detected checkpoint file, extracting model state dictionary")
            state_dict = state_dict['model_state_dict']
        
        # Check if the state_dict has DataParallel prefix 'module.' in keys
        if len(list(state_dict.keys())) > 0 and list(state_dict.keys())[0].startswith('module.'):
            print("Detected DataParallel model, removing 'module.' prefix for analysis")
            # Create new OrderedDict without the 'module.' prefix
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        # Detect number of classes from final layer bias
        if 'gcn.fc2.bias' in state_dict:
            num_classes = state_dict['gcn.fc2.bias'].shape[0]
            print(f"Detected {num_classes} classes in model")
        else:
            print("Could not detect number of classes, defaulting to 20")
            num_classes = 20
        
        # Detect image size from position embedding size
        if 'feature_extractor.encoder.pos_embedding' in state_dict:
            pos_embedding_size = state_dict['feature_extractor.encoder.pos_embedding'].shape
            if len(pos_embedding_size) == 3:
                num_patches = pos_embedding_size[1] - 1  # Subtract 1 for class token
                
                # Calculate patch size based on num_patches (assuming square patches)
                patch_size = int(np.sqrt(num_patches))
                
                # Typical ViT uses 16x16 patches
                patch_dim = 16
                img_size = patch_size * patch_dim
                
                print(f"Detected model trained with image size: {img_size}x{img_size} (patch grid: {patch_size}x{patch_size})")
                return img_size, num_classes
            
        # Default to 224 if we can't detect
        print("Could not detect image size from position embedding, defaulting to 224x224")
        return 224, num_classes
        
    except Exception as e:
        print(f"Error during model spec detection: {e}")
        print("Using defaults: 224x224 image size, 20 classes")
        return 224, 20

def initialize_transform(img_size):
    """
    Initialize the transform pipeline based on image size
    
    Args:
        img_size: Image size to use
    """
    global transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(f"Initialized transform with image size: {img_size}x{img_size}")

def load_model(model_path, num_classes=None, img_size=None):
    """
    Load a trained ViT backbone model
    
    Args:
        model_path: Path to saved model weights
        num_classes: Number of classes (will be detected if None)
        img_size: Image size for ViT model (will be detected if None)
        
    Returns:
        model: Loaded model
    """
    # Detect model specifications if not provided
    if img_size is None or num_classes is None:
        detected_img_size, detected_num_classes = detect_model_specs(model_path)
        img_size = img_size or detected_img_size
        num_classes = num_classes or detected_num_classes
    
    # Initialize transform with the detected image size
    initialize_transform(img_size)
    
    # Initialize model architecture with detected parameters
    print(f"Initializing ViT model with image size {img_size}x{img_size} and {num_classes} classes")
    model = ImageGCNSimple(
        num_classes=num_classes,
        feature_dim=768,  # ViT-Base feature dimension
        hidden_dim=512,
        dropout=0.0,  # No dropout needed for inference
        use_vit=True,  # Force ViT usage
        device=str(device),
        img_size=img_size
    )
    
    # Load weights
    print(f"Loading ViT model from {model_path}")
    try:
        # First try loading as is (normal model save)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        
        # Check if this is a checkpoint file with multiple dictionaries
        if 'model_state_dict' in state_dict:
            print("Detected checkpoint file, extracting model state dictionary")
            state_dict = state_dict['model_state_dict']
        
        # Check if the state_dict has DataParallel prefix 'module.' in keys
        is_data_parallel = False
        if len(list(state_dict.keys())) > 0:
            is_data_parallel = list(state_dict.keys())[0].startswith('module.')
            
        if is_data_parallel:
            print("Detected DataParallel model, removing 'module.' prefix")
            # Create new OrderedDict without the 'module.' prefix
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
                new_state_dict[name] = v
            state_dict = new_state_dict
            
        # Load state dict
        model.load_state_dict(state_dict)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using randomly initialized weights")
    
    model.to(device)
    model.eval()
    return model

def predict_tags(model, image, class_names, threshold=0.5):
    """
    Predict tags for an image
    
    Args:
        model: Trained model
        image: PIL Image
        class_names: List of class names
        threshold: Confidence threshold for predictions
        
    Returns:
        tags: Dictionary of class name -> confidence score
        top_tags: List of top tag names (above threshold)
    """
    # Transform image (transform must be initialized before calling this)
    if transform is None:
        raise ValueError("Transform not initialized. Call load_model first.")
        
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
    
    # Ensure probabilities is 1D even if batch size is 1
    if len(probabilities.shape) == 0:
        probabilities = probabilities.reshape(1)
    
    # Match the probabilities length with class_names length
    if len(probabilities) != len(class_names):
        print(f"Warning: Model output size ({len(probabilities)}) doesn't match number of class names ({len(class_names)})")
        # If model output is larger, truncate it
        if len(probabilities) > len(class_names):
            probabilities = probabilities[:len(class_names)]
        # If model output is smaller, pad with zeros
        else:
            probabilities = np.pad(probabilities, (0, len(class_names) - len(probabilities)))
    
    # Create tag dictionary
    tags = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    # Get tags above threshold
    top_tags = [class_names[i] for i in range(len(class_names)) if probabilities[i] >= threshold]
    
    return tags, top_tags

def get_class_names(dataset="voc", num_classes=None):
    """
    Get class names for the specified dataset
    
    Args:
        dataset: Dataset name ('voc' or 'coco')
        num_classes: Detected number of classes
        
    Returns:
        class_names: List of class names
    """
    voc_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # Auto-detect dataset based on number of classes if possible
    if num_classes is not None:
        if num_classes == 20:
            print(f"Detected VOC dataset (20 classes)")
            return voc_classes
        elif num_classes == 80:
            print(f"Detected COCO dataset (80 classes)")
            return coco_classes
    
    # Otherwise use the user-specified dataset
    if dataset.lower() == "voc":
        return voc_classes
    elif dataset.lower() == "coco":
        return coco_classes
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def tag_image_file(image_path, model_path, dataset="auto", threshold=0.5, img_size=None):
    """
    Tag an image file using the trained ViT model
    
    Args:
        image_path: Path to image file
        model_path: Path to model weights
        dataset: Dataset name for class names ('voc', 'coco', or 'auto')
        threshold: Confidence threshold
        img_size: Override the auto-detected image size
        
    Returns:
        image: Original image
        tags: Dictionary of class name -> confidence score
        top_tags: List of top tag names (above threshold)
    """
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, {}, []
    
    # Load model and detect specs
    model = load_model(model_path, img_size=img_size)
    
    # Get class names based on detected model specs
    if dataset.lower() == "auto":
        # Get num_classes from the model's output layer
        num_classes = model.gcn.fc2.out_features
        class_names = get_class_names(num_classes=num_classes)
    else:
        class_names = get_class_names(dataset)
    
    # Predict tags
    tags, top_tags = predict_tags(model, image, class_names, threshold)
    
    return image, tags, top_tags

def display_results(image, tags, top_tags):
    """Display results in terminal"""
    print("\nPredicted Tags:")
    for tag in sorted(top_tags):
        print(f"- {tag}: {tags[tag]:.4f}")
    
    if not top_tags:
        print("No tags predicted above threshold")

def gradio_interface(image, model_path, dataset, threshold, img_size):
    """
    Gradio interface function for image tagging with ViT
    
    Args:
        image: Input image
        model_path: Path to model
        dataset: Dataset for class names
        threshold: Confidence threshold
        img_size: Override the auto-detected image size
        
    Returns:
        result_image: Image with tags
        result_html: HTML with tag list
    """
    if image is None:
        return None, "Please upload an image"
    
    # Load model and detect specs
    model = load_model(model_path, img_size=img_size)
    
    # Get class names based on detected model specs
    if dataset.lower() == "auto":
        # Get num_classes from the model's output layer
        num_classes = model.gcn.fc2.out_features
        class_names = get_class_names(num_classes=num_classes)
    else:
        class_names = get_class_names(dataset)
    
    # Predict tags
    tags, top_tags = predict_tags(model, image, class_names, threshold)
    
    # Create result image with tags
    result_image = image.copy()
    
    # Generate HTML for results
    if not top_tags:
        result_html = "<div style='color: red'>No tags found above threshold.</div>"
    else:
        result_html = "<div style='text-align: left'><h3>Predicted Tags:</h3><ul>"
        for tag in sorted(top_tags):
            result_html += f"<li><b>{tag}</b>: {tags[tag]:.4f}</li>"
        result_html += "</ul></div>"
    
    return result_image, result_html

def launch_web_interface(model_path, dataset, threshold, img_size=None):
    """Launch the web interface for ViT model"""
    # Auto-detect image size and class count from model if not specified
    if img_size is None:
        detected_img_size, _ = detect_model_specs(model_path)
        img_size = detected_img_size
    
    # Create Gradio interface
    iface = gr.Interface(
        fn=lambda img: gradio_interface(img, model_path, dataset, threshold, img_size),
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Image(type="pil", label="Input Image"),
            gr.HTML(label="Predicted Tags")
        ],
        title="GCN Image Tagger (ViT Backbone)",
        description=(
            "Upload an image to get predicted tags using a trained GCN model with ViT backbone. "
            f"Using model: {os.path.basename(model_path)}, "
            f"dataset: {dataset if dataset != 'auto' else 'auto-detected'}, "
            f"threshold: {threshold}, "
            f"image size: {img_size}x{img_size}"
        ),
    )
    
    # Launch the interface
    iface.launch(share=True)

def main():
    parser = argparse.ArgumentParser(description='GCN Image Tagger with ViT Backbone')
    parser.add_argument('--image', type=str, help='Path to an image file')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--dataset', type=str, default='auto', choices=['voc', 'coco', 'auto'], 
                        help='Dataset for class names (auto will detect from model)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Confidence threshold for predictions')
    parser.add_argument('--img-size', type=int, default=None,
                        help='Override the auto-detected image size (e.g., 224, 384, 448)')
    parser.add_argument('--web', action='store_true', help='Launch web interface')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Launch web interface if requested or if no image provided
    if args.web or args.image is None:
        try:
            import gradio as gr
            print("Launching web interface for ViT model...")
            launch_web_interface(
                model_path=args.model,
                dataset=args.dataset,
                threshold=args.threshold,
                img_size=args.img_size
            )
        except ImportError:
            print("Error: Gradio not installed. Install it with 'pip install gradio'")
            print("Or provide an image path using --image")
    else:
        # Process single image
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            return
        
        # Tag the image
        image, tags, top_tags = tag_image_file(
            image_path=args.image,
            model_path=args.model,
            dataset=args.dataset,
            threshold=args.threshold,
            img_size=args.img_size
        )
        
        # Display results
        if image is not None:
            display_results(image, tags, top_tags)

if __name__ == "__main__":
    main() 