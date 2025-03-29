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

### Running the Project

After activating the virtual environment, you can run the project:

```
python test.py
```

### Deactivating the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:

```
deactivate
```

## Project Description

This project implements an ML-GCN (Multi-Label Graph Convolutional Network) model for image tagging. It uses ResNet-50 as the backbone for feature extraction and a Graph Convolutional Network to model label relationships.
