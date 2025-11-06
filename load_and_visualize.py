#!/usr/bin/env python3
"""
Quick script to load KeyBot models and visualize predictions
Usage: python load_and_visualize.py

Note: TensorBoard import warnings can be safely ignored.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
import warnings

# Suppress TensorBoard warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorboard')

# Add codes directory to path
import sys
sys.path.insert(0, './codes')

# Change to codes directory (required for relative paths in the code)
original_dir = os.getcwd()
codes_dir = os.path.join(original_dir, 'codes')
os.chdir(codes_dir)

from AnomalySuggestion_get_model import get_keypoint_model, get_test_data_loader
from suggest_codes.get_suggest_model import SuggestionConvModel
from suggest_codes.get_pseudo_generation_model_image_heatmap import PseudoLabelModel

# Change back to original directory
os.chdir(original_dir)

# ============================================================================
# 1. LOAD MODELS
# ============================================================================

print("Loading models...")

# Load Refiner (main model)
trainer, save_manager = get_keypoint_model(data='spineweb')
refiner = trainer.model
refiner.eval()

# Load Detector
detector = SuggestionConvModel()
detector_path = os.path.join(original_dir, 'save_suggestion', 'AASCE_suggestModel.pth')
if os.path.exists(detector_path):
    detector.load_state_dict(torch.load(
        detector_path,
        map_location='cuda' if torch.cuda.is_available() else 'cpu'
    ))
    detector.eval()
    print("✓ Detector loaded")
else:
    print(f"⚠ Detector model not found at {detector_path}")

# Load Corrector
corrector = PseudoLabelModel(n_keypoint=68, num_bones=17)
corrector_path = os.path.join(original_dir, 'save_refine', 'AASCE_refineModel.pth')
if os.path.exists(corrector_path):
    corrector.load_state_dict(torch.load(
        corrector_path,
        map_location='cuda' if torch.cuda.is_available() else 'cpu'
    ))
    corrector.eval()
    print("✓ Corrector loaded")
else:
    print(f"⚠ Corrector model not found at {corrector_path}")

print("✓ Models loaded successfully")

# ============================================================================
# 2. LOAD TEST DATA
# ============================================================================

print("Loading test data...")
_, _, test_loader = get_test_data_loader(data='spineweb')

# Get one batch
batch = next(iter(test_loader))
image = batch['input_image'][0]  # (3, 512, 256)
gt_coords = batch['label']['coord'][0]  # (68, 2)
image_path = batch['input_image_path'][0]

print(f"✓ Loaded image: {image_path}")
print(f"  Image shape: {image.shape}")
print(f"  Ground truth coords shape: {gt_coords.shape}")

# ============================================================================
# 3. RUN INFERENCE
# ============================================================================

print("\nRunning inference...")

with torch.no_grad():
    # Prepare batch
    batch['is_training'] = False
    batch['hint'] = {'index': [None]}  # No hints
    
    # Forward pass through Refiner
    out, batch = refiner(batch)
    
    # Get predicted coordinates (soft-argmax)
    pred_coords = out.pred.sargmax_coord[0].cpu()  # (68, 2)
    pred_heatmap = out.pred.heatmap[0].cpu()  # (68, 512, 256)

print("✓ Inference complete")
print(f"  Predicted coords shape: {pred_coords.shape}")
print(f"  Predicted heatmap shape: {pred_heatmap.shape}")

# ============================================================================
# 4. COMPUTE ERROR
# ============================================================================

# Mean Radial Error
mre = torch.sqrt(((pred_coords - gt_coords.cpu())**2).sum(-1)).mean()
print(f"\n📊 Mean Radial Error: {mre:.2f} pixels")

# ============================================================================
# 5. VISUALIZE RESULTS
# ============================================================================

print("\nCreating visualization...")

# Denormalize image for display
display_image = image.permute(1, 2, 0).cpu().numpy()  # (512, 256, 3)
display_image = (display_image + 1) / 2  # [-1, 1] -> [0, 1]
display_image = np.clip(display_image, 0, 1)

# Convert to grayscale for better visibility
display_image_gray = display_image.mean(axis=2)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# ---- Row 1: Images ----

# 1. Original image with ground truth
ax = axes[0, 0]
ax.imshow(display_image_gray, cmap='gray')
ax.scatter(gt_coords[:, 1], gt_coords[:, 0], c='green', s=20, alpha=0.7, label='Ground Truth')
ax.set_title('Original + Ground Truth', fontsize=12)
ax.axis('off')
ax.legend()

# 2. Original image with predictions
ax = axes[0, 1]
ax.imshow(display_image_gray, cmap='gray')
ax.scatter(pred_coords[:, 1], pred_coords[:, 0], c='red', s=20, alpha=0.7, label='Predictions')
ax.set_title('Original + Predictions', fontsize=12)
ax.axis('off')
ax.legend()

# 3. Overlay: GT + Predictions
ax = axes[0, 2]
ax.imshow(display_image_gray, cmap='gray')
ax.scatter(gt_coords[:, 1], gt_coords[:, 0], c='green', s=30, alpha=0.6, label='GT', marker='o')
ax.scatter(pred_coords[:, 1], pred_coords[:, 0], c='red', s=20, alpha=0.6, label='Pred', marker='x')
# Draw lines connecting GT to Pred
for i in range(len(gt_coords)):
    ax.plot([gt_coords[i, 1], pred_coords[i, 1]], 
            [gt_coords[i, 0], pred_coords[i, 0]], 
            'yellow', linewidth=0.5, alpha=0.3)
ax.set_title(f'Comparison (MRE: {mre:.2f}px)', fontsize=12)
ax.axis('off')
ax.legend()

# ---- Row 2: Heatmaps ----

# 4. Sum of all GT heatmaps
ax = axes[1, 0]
gt_heatmap = batch['label']['heatmap'][0].cpu().sum(dim=0)  # Sum over 68 keypoints
ax.imshow(display_image_gray, cmap='gray', alpha=0.5)
ax.imshow(gt_heatmap, cmap='hot', alpha=0.5)
ax.set_title('Ground Truth Heatmaps', fontsize=12)
ax.axis('off')

# 5. Sum of all predicted heatmaps
ax = axes[1, 1]
pred_heatmap_sum = pred_heatmap.sum(dim=0)  # Sum over 68 keypoints
ax.imshow(display_image_gray, cmap='gray', alpha=0.5)
ax.imshow(pred_heatmap_sum, cmap='hot', alpha=0.5)
ax.set_title('Predicted Heatmaps', fontsize=12)
ax.axis('off')

# 6. Error map (per-keypoint error)
ax = axes[1, 2]
errors = torch.sqrt(((pred_coords - gt_coords.cpu())**2).sum(-1))  # (68,)
# Create error visualization
error_image = np.zeros((512, 256))
for i, (coord, error) in enumerate(zip(pred_coords, errors)):
    row, col = int(coord[0]), int(coord[1])
    if 0 <= row < 512 and 0 <= col < 256:
        # Draw circle with size proportional to error
        y, x = np.ogrid[-10:10, -10:10]
        mask = x**2 + y**2 <= (error * 2)**2
        r_min, r_max = max(0, row-10), min(512, row+10)
        c_min, c_max = max(0, col-10), min(256, col+10)
        error_image[r_min:r_max, c_min:c_max] = np.maximum(
            error_image[r_min:r_max, c_min:c_max],
            mask[:r_max-r_min, :c_max-c_min] * error.item()
        )
ax.imshow(display_image_gray, cmap='gray', alpha=0.5)
im = ax.imshow(error_image, cmap='Reds', alpha=0.6, vmin=0, vmax=errors.max())
ax.set_title('Error Map (Red = High Error)', fontsize=12)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
output_image_path = os.path.join(original_dir, 'keybot_visualization.png')
plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to '{output_image_path}'")
plt.show()

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================

# Save coordinates to JSON
results = {
    'image_path': image_path,
    'mean_radial_error_pixels': float(mre),
    'ground_truth_coords': gt_coords.cpu().tolist(),
    'predicted_coords': pred_coords.tolist(),
    'per_keypoint_errors': torch.sqrt(((pred_coords - gt_coords.cpu())**2).sum(-1)).tolist()
}

output_json_path = os.path.join(original_dir, 'keybot_results.json')
with open(output_json_path, 'w') as f:
    json.dump(results, f, indent=2)
    
print(f"✓ Saved results to '{output_json_path}'")

# ============================================================================
# 7. OPTIONAL: Run with KeyBot (Detector + Corrector)
# ============================================================================

print("\n" + "="*60)
print("OPTIONAL: Running with KeyBot auto-correction...")
print("="*60)

# This would require implementing the full iteration loop
# See evaluate_AASCE.py for complete implementation
print("(See evaluate_AASCE.py for full KeyBot collaborative inference)")

print("\n✅ Done! Check:")
print(f"  - {output_image_path}")
print(f"  - {output_json_path}")

