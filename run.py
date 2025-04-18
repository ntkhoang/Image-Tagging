import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr

from gcn_image_tagger_simplified import ImageGCNSimple

# Set the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms will be updated based on image_size parameter
transform = None

def load_model(model_path, num_classes=20, use_vit=False, image_size=224):
    """
    Load a trained model
    
    Args:
        model_path: Path to saved model weights
        num_classes: Number of classes
        use_vit: Whether to use ViT as backbone
        image_size: Image size for ViT model (224, 384, 448, etc.)
        
    Returns:
        model: Loaded model
    """
    # Initialize model architecture
    feature_dim = 768 if use_vit else 2048  # ViT-Base vs ResNet-50
    model = ImageGCNSimple(
        num_classes=num_classes,
        feature_dim=feature_dim,
        hidden_dim=512,
        dropout=0.0,  # No dropout needed for inference
        use_vit=use_vit,
        device=str(device),
        img_size=image_size if use_vit else None  # Only pass img_size for ViT models
    )
    
    # Load weights
    print(f"Loading model from {model_path}")
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
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
                new_state_dict[name] = v
            state_dict = new_state_dict
            
        model.load_state_dict(state_dict)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using randomly initialized weights")
    
    model.to(device)
    model.eval()
    return model

def initialize_transform(image_size=224):
    """Initialize the transform with the specified image size"""
    global transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(f"Initialized transform with image size: {image_size}x{image_size}")

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
    # Transform image (using global transform)
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
    
    # Create tag dictionary
    tags = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    # Get tags above threshold
    top_tags = [class_names[i] for i in range(len(class_names)) if probabilities[i] >= threshold]
    
    return tags, top_tags

def get_class_names(dataset="voc"):
    """
    Get class names for the specified dataset
    
    Args:
        dataset: Dataset name ('voc' or 'coco')
        
    Returns:
        class_names: List of class names
    """
    if dataset.lower() == "voc":
        return [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
    elif dataset.lower() == "coco":
        # Simplified list of common COCO classes
        # In a real application, you'd load the full 80 classes
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack'
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def tag_image_file(image_path, model_path, dataset="voc", threshold=0.5, use_vit=False, image_size=224):
    """
    Tag an image file using the trained model
    
    Args:
        image_path: Path to image file
        model_path: Path to model weights
        dataset: Dataset name for class names ('voc' or 'coco')
        threshold: Confidence threshold
        use_vit: Whether the model uses ViT backbone
        image_size: Image size for the model (especially important for ViT)
        
    Returns:
        image: Original image
        tags: Dictionary of class name -> confidence score
        top_tags: List of top tag names (above threshold)
    """
    # Initialize transform with the right image size
    initialize_transform(image_size)
    
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, {}, []
    
    # Get class names
    class_names = get_class_names(dataset)
    
    # Load model
    model = load_model(
        model_path=model_path,
        num_classes=len(class_names),
        use_vit=use_vit,
        image_size=image_size
    )
    
    # Predict tags
    tags, top_tags = predict_tags(model, image, class_names, threshold)
    
    return image, tags, top_tags

def display_results(image, tags, top_tags):
    """Display results in terminal"""
    print("\nPredicted Tags:")
    for tag in top_tags:
        print(f"- {tag}: {tags[tag]:.4f}")
    
    if not top_tags:
        print("No tags predicted above threshold")

def gradio_interface(image, model_path, dataset, threshold, use_vit, image_size):
    """
    Gradio interface function for image tagging
    
    Args:
        image: Input image
        model_path: Path to model
        dataset: Dataset for class names
        threshold: Confidence threshold
        use_vit: Whether to use ViT backbone
        image_size: Image size for the model
        
    Returns:
        result_image: Image with tags
        result_html: HTML with tag list
    """
    if image is None:
        return None, "Please upload an image"
    
    # Initialize transform with the right image size
    initialize_transform(image_size)
    
    # Get class names
    try:
        class_names = get_class_names(dataset)
    except ValueError:
        return None, "Unknown dataset. Please use 'voc' or 'coco'."
    
    # Load model
    model = load_model(
        model_path=model_path,
        num_classes=len(class_names),
        use_vit=use_vit,
        image_size=image_size
    )
    
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

def launch_web_interface(model_path, dataset, threshold, use_vit, image_size):
    """Launch the web interface"""
    # Initialize transform with the right image size
    initialize_transform(image_size)
    
    # Create Gradio interface
    iface = gr.Interface(
        fn=lambda img: gradio_interface(img, model_path, dataset, threshold, use_vit, image_size),
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Image(type="pil", label="Input Image"),
            gr.HTML(label="Predicted Tags")
        ],
        title="GCN Image Tagger",
        description=(
            "Upload an image to get predicted tags using a trained GCN model. "
            f"Using model: {os.path.basename(model_path)}, "
            f"dataset: {dataset}, threshold: {threshold}, "
            f"{'ViT backbone' if use_vit else 'ResNet backbone'}, "
            f"image size: {image_size}x{image_size}"
        ),
    )
    
    # Launch the interface
    iface.launch(share=True)

def main():
    parser = argparse.ArgumentParser(description='GCN Image Tagger')
    parser.add_argument('--image', type=str, help='Path to an image file')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--dataset', type=str, default='voc', choices=['voc', 'coco'], 
                        help='Dataset for class names')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Confidence threshold for predictions')
    parser.add_argument('--use-vit', action='store_true', help='Use ViT backbone')
    parser.add_argument('--image-size', type=int, default=224, 
                        help='Image size for model input (224, 384, 448, etc.)')
    parser.add_argument('--web', action='store_true', help='Launch web interface')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Initialize the transform with the specified image size
    initialize_transform(args.image_size)
    
    # Launch web interface if requested or if no image provided
    if args.web or args.image is None:
        try:
            import gradio as gr
            print("Launching web interface...")
            launch_web_interface(
                model_path=args.model,
                dataset=args.dataset,
                threshold=args.threshold,
                use_vit=args.use_vit,
                image_size=args.image_size
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
            use_vit=args.use_vit,
            image_size=args.image_size
        )
        
        # Display results
        if image is not None:
            display_results(image, tags, top_tags)

if __name__ == "__main__":
    main() 