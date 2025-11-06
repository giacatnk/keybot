# KeyBot - Google Colab Usage Guide

## 🚀 Quick Start on Google Colab

### Step 1: Clone Repository

```python
# In a Colab code cell:
!git clone https://github.com/giacatnk/keybot.git
%cd keybot
```

### Step 2: Setup Environment

```bash
# Auto-detects Colab and uses pip instead of poetry
!bash setup_and_run.sh setup
```

**What it does:**
- Detects Google Colab environment
- Installs dependencies via pip (not poetry)
- Verifies PyTorch, CUDA, and other packages

### Step 3: Prepare Dataset

**Option A: Upload to Colab** (for quick testing)
```python
from google.colab import files
import zipfile

# Upload your dataset zip file
uploaded = files.upload()

# Extract
!unzip dataset.zip -d codes/preprocess_data/AASCE_rawdata/boostnet_labeldata/
```

**Option B: Mount Google Drive** (recommended)
```python
from google.colab import drive
drive.mount('/content/drive')

# Create symlink to your dataset in Drive
!ln -s /content/drive/MyDrive/keybot_data/AASCE_rawdata codes/preprocess_data/
```

Then run preprocessing:
```bash
!bash setup_and_run.sh data
```

### Step 4: Upload or Train Models

**Option A: Upload Pre-trained Models** (fastest)
```python
from google.colab import files

# Upload models
# save/AASCE_interactive_keypoint_estimation/model.pth
# save_suggestion/AASCE_suggestModel.pth  
# save_refine/AASCE_refineModel.pth
```

**Option B: Train from Scratch** (~30 hours on GPU)
```bash
!bash setup_and_run.sh train
```

### Step 5: Run Inference & Visualization

```python
!python3 load_and_visualize.py
```

**View Results:**
```python
from IPython.display import Image, display
display(Image('keybot_visualization.png'))

import json
with open('keybot_results.json') as f:
    results = json.load(f)
    print(f"Mean Radial Error: {results['mean_radial_error_pixels']:.2f} pixels")
```

### Step 6: Run Full Evaluation

```bash
!bash setup_and_run.sh eval
```

---

## 📋 All Available Commands

```bash
# Setup and installation
!bash setup_and_run.sh setup      # Install dependencies
!bash setup_and_run.sh verify     # Verify installation

# Data preparation
!bash setup_and_run.sh data       # Preprocess dataset

# Training (long-running)
!bash setup_and_run.sh train      # Train all models (~30 hours)

# Evaluation
!bash setup_and_run.sh eval       # Run evaluation on test set
!bash setup_and_run.sh results    # Show training results

# All in one (setup + data + train + eval)
!bash setup_and_run.sh all        # Run everything
```

---

## 💾 Saving Results to Google Drive

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy results
!cp -r save/ /content/drive/MyDrive/keybot_results/
!cp keybot_visualization.png /content/drive/MyDrive/keybot_results/
!cp keybot_results.json /content/drive/MyDrive/keybot_results/
```

---

## 🔧 GPU Configuration

Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

Set GPU device:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
```

---

## 📊 Monitoring Training (TensorBoard)

```python
# Load TensorBoard extension
%load_ext tensorboard

# Start TensorBoard
%tensorboard --logdir save/AASCE_interactive_keypoint_estimation/
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: Out of Memory (OOM)
```python
# Reduce batch size in config
!sed -i 's/batch_size: 4/batch_size: 2/g' codes/config/config_AASCE.yaml
```

### Issue 2: Session Timeout (Training > 12 hours)
```python
# Use colab_keepalive.py to prevent disconnection
!python colab_keepalive.py &

# Or split training into stages
!python3 codes/main.py --config config_AASCE  # Refiner only
# Save, restart session, then:
!python3 codes/train_AASCE.py  # Detector + Corrector
```

### Issue 3: Dataset Not Found
```bash
# Verify dataset structure
!ls -la codes/preprocess_data/AASCE_rawdata/boostnet_labeldata/
# Should contain: data/ and labels/ directories
```

### Issue 4: Module Not Found
```python
# Reinstall dependencies
!pip install -q torch torchvision albumentations scipy munch pytz tqdm opencv-python pandas matplotlib
```

---

## 🎯 Example: Complete Workflow

```python
# 1. Setup
!git clone https://github.com/giacatnk/keybot.git
%cd keybot
!bash setup_and_run.sh setup

# 2. Mount Drive and link dataset
from google.colab import drive
drive.mount('/content/drive')
!ln -s /content/drive/MyDrive/keybot_data/AASCE_rawdata codes/preprocess_data/

# 3. Preprocess
!bash setup_and_run.sh data

# 4. Upload pre-trained models (or train)
# ... upload files via Colab UI ...

# 5. Run visualization
!python3 load_and_visualize.py

# 6. Display results
from IPython.display import Image
display(Image('keybot_visualization.png'))
```

---

## 📝 Notes

- **Colab Free**: 12-hour session limit, may disconnect
- **Colab Pro**: 24-hour sessions, better GPUs
- **Training Time**: 
  - Refiner: ~24 hours
  - Detector: ~3 hours
  - Corrector: ~5 hours
  - Total: ~30 hours (use checkpoints!)
  
- **Storage**: 
  - Dataset: ~2 GB
  - Models: ~500 MB total
  - Use Google Drive for persistence

---

## 🔗 Useful Links

- **Dataset**: http://spineweb.digitalimaginggroup.ca/Index.php?n=Main.Datasets#Dataset_16.3A_609_spinal_anterior-posterior_x-ray_images
- **Paper**: ECCV 2024
- **GitHub**: https://github.com/giacatnk/keybot

---

**Ready to run KeyBot on Google Colab! 🎉**

