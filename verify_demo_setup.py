#!/usr/bin/env python3
"""
Verification script to check if everything is ready for the demo notebook.
Run this before opening the notebook to ensure all dependencies are met.

Usage: python verify_demo_setup.py
"""

import os
import sys
from pathlib import Path

def check_item(name, condition, error_msg=""):
    """Check a single item and print status"""
    status = "✓" if condition else "✗"
    color = "\033[92m" if condition else "\033[91m"  # Green or Red
    reset = "\033[0m"
    
    print(f"{color}{status}{reset} {name}")
    if not condition and error_msg:
        print(f"  → {error_msg}")
    return condition

def main():
    print("="*60)
    print("KeyBot Demo Notebook - Setup Verification")
    print("="*60)
    print()
    
    all_good = True
    
    # Check Python packages
    print("📦 Checking Python Packages...")
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'PIL': 'Pillow (PIL)',
        'tqdm': 'tqdm',
    }
    
    for module, name in packages.items():
        try:
            __import__(module)
            check_item(f"{name} installed", True)
        except ImportError:
            all_good = False
            check_item(f"{name} installed", False, f"Install with: pip install {module if module != 'PIL' else 'Pillow'}")
    
    print()
    
    # Check files and directories
    print("📁 Checking Files and Directories...")
    
    # Check notebook
    all_good &= check_item(
        "Demo notebook exists",
        os.path.exists("demo_notebook.ipynb"),
        "Run the notebook creation script first"
    )
    
    # Check codes directory
    all_good &= check_item(
        "Codes directory exists",
        os.path.exists("codes"),
        "Missing codes directory"
    )
    
    # Check model files
    print()
    print("🤖 Checking Pre-trained Models...")
    
    refiner_path = "save/AASCE_interactive_keypoint_estimation/model.pth"
    all_good &= check_item(
        "Refiner model (main)",
        os.path.exists(refiner_path),
        f"Missing: {refiner_path}"
    )
    
    detector_path = "save_suggestion/AASCE_suggestModel.pth"
    detector_exists = os.path.exists(detector_path)
    check_item(
        "Detector model (optional)",
        detector_exists,
        f"Missing: {detector_path} (notebook will still work)"
    )
    
    corrector_path = "save_refine/AASCE_refineModel.pth"
    corrector_exists = os.path.exists(corrector_path)
    check_item(
        "Corrector model (optional)",
        corrector_exists,
        f"Missing: {corrector_path} (notebook will still work)"
    )
    
    print()
    print("📊 Checking Test Data...")
    
    # Check preprocessed data
    test_data_path = "codes/preprocess_data/AASCE_processed/test/"
    test_json_path = "codes/preprocess_data/AASCE_processed/test.json"
    
    test_data_exists = os.path.exists(test_data_path) and os.path.exists(test_json_path)
    all_good &= check_item(
        "Test data preprocessed",
        test_data_exists,
        "Run: cd codes/preprocess_data && python data_split.py && python preprocess_data.py"
    )
    
    if test_data_exists:
        # Count test samples
        import json
        try:
            with open(test_json_path, 'r') as f:
                test_data = json.load(f)
            num_samples = len(test_data)
            check_item(
                f"Test samples available: {num_samples}",
                num_samples > 0
            )
        except Exception as e:
            check_item("Test data readable", False, str(e))
    
    print()
    print("🔧 Checking System...")
    
    # Check PyTorch and CUDA
    try:
        import torch
        check_item("PyTorch imported", True)
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            check_item(f"CUDA available: {torch.cuda.get_device_name(0)}", True)
        else:
            check_item("CUDA available (CPU mode will be used)", False, 
                      "GPU not available, will use CPU (slower but works)")
    except Exception as e:
        all_good = False
        check_item("PyTorch", False, str(e))
    
    # Check Jupyter
    print()
    print("📓 Checking Jupyter...")
    
    try:
        import notebook
        check_item("Jupyter Notebook installed", True)
    except ImportError:
        try:
            import jupyterlab
            check_item("JupyterLab installed", True)
        except ImportError:
            print("  ⚠ Neither Jupyter Notebook nor JupyterLab found")
            print("  → Install with: pip install jupyter")
    
    # Summary
    print()
    print("="*60)
    if all_good:
        print("✅ All checks passed! Ready to run the demo notebook.")
        print()
        print("To start the demo:")
        print("  jupyter notebook demo_notebook.ipynb")
        print()
        print("Or with JupyterLab:")
        print("  jupyter lab demo_notebook.ipynb")
    else:
        print("⚠️  Some issues found. Please fix them before running the demo.")
        print()
        print("Quick fixes:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Ensure models are trained or downloaded")
        print("  3. Preprocess data if not done yet")
    print("="*60)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())

