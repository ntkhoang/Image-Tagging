# GCN Image Tagger

A Graph Convolutional Network (GCN) based image tagging tool that identifies objects in images.

## Features

- Multi-label image classification using GCN
- Pre-trained feature extractors (ResNet-50, ViT)
- Command-line interface for batch processing
- Web interface for interactive use
- Support for VOC and COCO datasets

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Image-Tagging.git
   cd Image-Tagging
   ```

2. Install required packages:
   ```
   pip install torch torchvision pillow numpy tqdm
   ```

3. For the web interface, install Gradio:
   ```
   pip install gradio
   ```

## Usage

### Command-line Interface

To tag a single image:

```bash
python run.py --image path/to/your/image.jpg --model path/to/your/model.pth
```

Options:
- `--image`: Path to input image (required)
- `--model`: Path to trained model weights (required)
- `--dataset`: Dataset for class names (`voc` or `coco`, default: `voc`)
- `--threshold`: Confidence threshold for predictions (default: 0.5)
- `--use-vit`: Use ViT backbone (default: False)

### Web Interface

To launch the web interface:

```bash
python run.py --model path/to/your/model.pth --web
```

Or simply omit the `--image` parameter to automatically launch the web interface:

```bash
python run.py --model path/to/your/model.pth
```

The web interface allows you to:
- Upload images from your computer
- See prediction results instantly
- Adjust confidence threshold
- View confidence scores for each detected class

## Training Your Own Models

Refer to our training scripts for training on your own data:

- For VOC dataset: `train_voc.py`
- For COCO dataset: `train_model_simplified.py`
- For evaluating models: `evaluate_voc.py` or `evaluate_coco_simplified.py`

Example training command:

```bash
python train_voc.py --voc-dir /path/to/VOCdevkit/VOC2007 --batch-size 16 --epochs 30 --save-dir ./models
```

## Model Architecture

Our model combines:
- Pre-trained backbone (ResNet-50 or Vision Transformer)
- Graph Convolutional Network for modeling label relationships
- Multi-label classification head

## Example

```bash
# Tag an image with VOC classes
python run.py --image examples/dog.jpg --model models/best_model_voc.pth --dataset voc

# Tag an image with COCO classes
python run.py --image examples/person.jpg --model models/best_model_coco.pth --dataset coco --threshold 0.3

# Launch web interface
python run.py --model models/best_model_voc.pth --web
```

## Adding Your Own Images to Examples

You can add your own example images to the `examples/` directory:

```
examples/
  ├── dog.jpg
  ├── cat.jpg
  ├── car.jpg
  └── your_image.jpg
```

These will automatically appear in the web interface as example images.
