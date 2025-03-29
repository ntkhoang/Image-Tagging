# Image Tagging Project

## Setup with uv

This project uses `uv` for Python package management and virtual environments. `uv` is a fast, user-friendly alternative to pip and venv.

### Installing uv

If you don't have uv installed, you can install it:

**Windows (PowerShell):**
```powershell
iwr -Uri https://github.com/astral-sh/uv/releases/latest/download/uv-installer.ps1 -UseBasicParsing | iex
```

**macOS/Linux:**
```bash
curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | sh
```

### Creating a Virtual Environment

1. Create a new virtual environment in the project directory:
   ```
   uv venv
   ```
   This will create a `.venv` directory in your project.

2. Activate the virtual environment:

   **Windows (PowerShell):**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   **Windows (Command Prompt):**
   ```cmd
   .\.venv\Scripts\activate.bat
   ```

   **macOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies from requirements.txt:
   ```
   uv pip install -r requirements.txt
   ```

## Using MS COCO Dataset

### Download COCO Dataset

1. Download the COCO 2017 dataset:
   - [Train images (18GB)](http://images.cocodataset.org/zips/train2017.zip)
   - [Validation images (1GB)](http://images.cocodataset.org/zips/val2017.zip)
   - [Annotations (241MB)](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

2. Extract these files to create the following directory structure:
   ```
   /path/to/coco/
   ├── annotations/
   │   ├── instances_train2017.json
   │   └── instances_val2017.json
   ├── train2017/
   │   ├── 000000000001.jpg
   │   └── ...
   └── val2017/
       ├── 000000000001.jpg
       └── ...
   ```

### Creating Adjacency Matrix

For ML-GCN to work, you need to create an adjacency matrix based on label co-occurrences:

1. Run the preprocessing script to generate the adjacency matrix:
   ```
   python generate_adj.py --data /path/to/coco/
   ```
   This will create a file at `data/adj_matrix.pkl`.

2. If you don't have the preprocessing script, you can also download a pre-computed adjacency matrix from:
   https://github.com/Megvii-Nanjing/ML-GCN/tree/master/data

### Running the Training

Run the training with:
```
python main.py --data /path/to/coco/ --batch-size 16 --image-size 448
```

For evaluation only:
```
python main.py --data /path/to/coco/ --evaluate --resume checkpoint/model_best.pth.tar
```

## Project Description

This project implements an ML-GCN (Multi-Label Graph Convolutional Network) model for image tagging. It uses ResNet-101 as the backbone for feature extraction and a Graph Convolutional Network to model label relationships.
