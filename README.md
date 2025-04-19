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

Pretrained models: [Google Drive](https://drive.google.com/drive/folders/1y2Yy30VyQyWulFgV3vzQnJ6rNxE13X_d?usp=sharing)

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
- `--image-size`: Image size for model input (default: 224, recommended: 448 for ViT models)

### Dedicated Scripts for Different Model Types

For easier use, we now provide dedicated scripts for each backbone type:

#### For ResNet Models
```bash
python run_resnet.py --image path/to/your/image.jpg --model path/to/your/resnet_model.pth
```

#### For ViT Models 
```bash
python run_vit.py --image path/to/your/image.jpg --model path/to/your/vit_model.pth
```

These specialized scripts handle the backbone-specific settings automatically (image size, feature dimensions, etc).

### Web Interface

To launch the web interface:

```bash
python run.py --model path/to/your/model.pth --web
```

Or simply omit the `--image` parameter to automatically launch the web interface:

```bash
python run.py --model path/to/your/model.pth
```

#### Using ViT Models in Web Interface

To use Vision Transformer (ViT) models with the web interface, you must specify both the `--use-vit` flag and the appropriate `--image-size` parameter:

```bash
python run.py --model path/to/your/vit_model.pth --use-vit --image-size 448 --web
```

Alternatively, use the dedicated ViT script which handles these settings automatically:

```bash
python run_vit.py --model path/to/your/vit_model.pth --web
```

> **Important:** ViT models must use the same image size for inference as was used during training. Common image sizes for ViT models are 224, 384, and 448.

The web interface allows you to:
- Upload images from your computer
- See prediction results instantly
- Adjust confidence threshold
- View confidence scores for each detected class

## Training Your Own Models

Refer to our training scripts for training on your own data:

- For VOC dataset: `train_voc.py` or `train_voc_ViT.py`
- For COCO dataset: `train_coco.py` or `train_coco_ViT.py`
- For evaluating models: `evaluate_voc.py` or `evaluate_coco.py`
- For evaluating ViT models: `evaluate_voc_ViT.py` or `evaluate_coco_ViT.py`

Example training command:

For ResNet backbone:

```bash
python train_voc.py --voc-dir /path/to/VOCdevkit/VOC2007 --batch-size 16 --epochs 30 --save-dir ./models
```

For ViT backbone:

```bash
python train_voc_ViT.py --voc-dir /kaggle/input/pascal-voc-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007 --batch-size 32 --epochs 30 --device cuda --image-size 448 --save-dir ./models
```

Example evaluating command:

```bash
python evaluate_coco.py --coco-dir /path/to/COCO2017 --model-path /pretrained/model/path --device cuda
```

For ViT models evaluation:
```bash
python evaluate_coco_ViT.py --coco-dir /path/to/COCO2017 --model-path /pretrained/vit/model/path --device cuda --image-size 448
```

## Model Architecture

Our model combines:
- Pre-trained backbone (ResNet-50 or Vision Transformer)
- Graph Convolutional Network for modeling label relationships
- Multi-label classification head

## Example

```bash
# Launch web interface with ResNet model
python run_resnet.py --model models/best_model_voc.pth --web

# Launch web interface with ViT model
python run_vit.py --model models/best_model_voc_vit.pth --img-size 448 --web
```
