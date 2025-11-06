# KeyBot System Components Overview

## 🏗️ SYSTEM COMPONENTS (3 Models + Data Processing Pipeline)

---

## 0. IMAGE PREPROCESSING PIPELINE

### **Raw Data Format**
- **Input**: Grayscale JPEG X-ray images + MATLAB .mat label files
- **Original Size**: Variable (1500-2400 × 900-1500 pixels)
- **Labels**: 68 (x, y) coordinates in original image space

### **Preprocessing Steps**

#### **Step 1: Data Organization**
```
Raw Data:
├─ Images: .jpg files (grayscale)
└─ Labels: .mat files (68 × 2 coordinates)
    ↓
Processed Data:
├─ Images: .npy arrays (RGB)
└─ Labels: .json files (coordinates + metadata)
```

#### **Step 2: Image Transformation**
```python
1. Grayscale to RGB:
   - Load grayscale image (H × W)
   - Repeat channel 3 times: (H × W) → (H × W × 3)
   
2. First Resize (Standardization):
   - Original (variable) → 1024 × 512
   - Maintains aspect ratio approximately
   
3. Final Resize (Model Input):
   - 1024 × 512 → 512 × 256
   - Both image AND keypoints scaled accordingly
   
4. Normalization:
   - Pixel values: [0, 255] → [0, 1] → [-1, 1]
   - Formula: (pixel / 255.0) * 2 - 1
```

#### **Step 3: Keypoint Transformation**
```python
# Keypoints must be transformed with same scaling as image
original_coords = load_from_mat_file()  # (68, 2) in original size

# Scale to match image resizing
scale_row = 512 / original_height
scale_col = 256 / original_width

transformed_coords[:, 0] = original_coords[:, 0] * scale_row
transformed_coords[:, 1] = original_coords[:, 1] * scale_col

# Result: (68, 2) coordinates in 512×256 space
```

#### **Step 4: Data Augmentation (Training Only)**
```python
# Using Albumentations library
Augmentations (applied to BOTH image and keypoints):
1. SafeRotate: ±15 degrees (p=0.5)
2. HorizontalFlip: Mirror image (p=0.5)
3. RandomScale: 0.9-1.2× (p=0.5)
4. RandomBrightnessContrast: Adjust intensity (p=default)
5. Final Resize: Ensure 512×256 (p=1.0)

KEY: Same transform applied to image AND coordinates
```

### **Heatmap Generation (From Coordinates)**

```python
def coord_to_heatmap(coords, image_size=(512, 256), std=7.5):
    """
    Convert keypoint coordinates to Gaussian heatmaps
    
    Args:
        coords: (68, 2) tensor of (row, col) coordinates
        image_size: (height, width) of output heatmap
        std: standard deviation of Gaussian (controls blob size)
    
    Returns:
        heatmaps: (68, 512, 256) tensor
    """
    heatmaps = []
    for coord in coords:  # For each of 68 keypoints
        # Create 2D Gaussian centered at coord
        mean = coord  # (row, col)
        variance = std ** 2
        
        # Create grid of (row, col) positions
        grid_row = torch.arange(512)
        grid_col = torch.arange(256)
        grid = torch.meshgrid(grid_row, grid_col)  # (512, 256, 2)
        
        # Gaussian formula: exp(-0.5 * (x-μ)²/σ²)
        diff = grid - mean
        gaussian = torch.exp(-0.5 * (diff**2).sum(-1) / variance)
        
        heatmaps.append(gaussian)
    
    return torch.stack(heatmaps)  # (68, 512, 256)
```

**Heatmap Properties**:
- **Size**: Same as image (512 × 256)
- **Range**: [0, 1] (Gaussian peak at 1.0)
- **Standard Deviation**: 7.5 pixels (blob size)
- **Purpose**: Probabilistic representation of keypoint location

### **Complete Processing Flow**

```
┌─────────────────────────────────────────────────────────┐
│ RAW DATA                                                │
│ X-ray.jpg (grayscale, 2000×1500) + labels.mat (68×2)  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Convert to RGB                                 │
│ (H, W) → (H, W, 3)                                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Resize to 1024×512                            │
│ Image: (2000, 1500, 3) → (1024, 512, 3)               │
│ Coords: (68, 2) × [1024/2000, 512/1500]               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Resize to 512×256 (Model Input Size)          │
│ Image: (1024, 512, 3) → (512, 256, 3)                 │
│ Coords: (68, 2) × [512/1024, 256/512]                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Normalize                                      │
│ Pixels: [0, 255] → [-1, 1]                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼ (Training only)
┌─────────────────────────────────────────────────────────┐
│ STEP 5: Data Augmentation                             │
│ - Rotate ±15°                                          │
│ - Flip horizontally                                    │
│ - Scale 0.9-1.2×                                       │
│ - Adjust brightness/contrast                          │
│ - Apply SAME transforms to image AND coordinates      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 6: Generate Heatmaps                             │
│ Coords (68, 2) → Heatmaps (68, 512, 256)              │
│ Gaussian blobs centered at each coordinate            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ FINAL BATCH                                            │
│ - Image: (B, 3, 512, 256) tensor, range [-1, 1]       │
│ - Coordinates: (B, 68, 2) tensor, range [0, 512]×[0, 256] │
│ - Heatmaps: (B, 68, 512, 256) tensor, range [0, 1]    │
│ - Ready for model input                                │
└─────────────────────────────────────────────────────────┘
```

### **Coordinate to Heatmap (Training)**
- **When**: During training, to create target heatmaps for loss computation
- **Input**: Ground truth coordinates (68, 2)
- **Output**: Ground truth heatmaps (68, 512, 256)
- **Used by**: All three models for supervision

### **Heatmap to Coordinate (Inference)**

```python
# Method 1: Hard-Argmax (Integer coordinates)
def hard_argmax(heatmap):
    """
    Args: heatmap (512, 256)
    Returns: coord (2,) - (row, col)
    """
    flat_idx = heatmap.argmax()
    row = flat_idx // 256
    col = flat_idx % 256
    return (row, col)

# Method 2: Soft-Argmax (Sub-pixel precision)
def soft_argmax(heatmap):
    """
    Args: heatmap (512, 256)
    Returns: coord (2,) - (row, col) with decimals
    """
    # Marginal distributions
    prob_row = heatmap.sum(dim=1)  # (512,)
    prob_col = heatmap.sum(dim=0)  # (256,)
    
    # Weighted average
    row_coords = torch.arange(512)
    col_coords = torch.arange(256)
    
    row = (prob_row * row_coords).sum() / prob_row.sum()
    col = (prob_col * col_coords).sum() / prob_col.sum()
    
    return (row, col)

# Method 3: Sub-pixel Refinement (Best precision)
def subpixel_argmax(heatmap, patch_size=15):
    """
    Args: heatmap (512, 256)
    Returns: coord (2,) - refined sub-pixel
    """
    # Step 1: Hard-argmax
    hard_coord = hard_argmax(heatmap)
    
    # Step 2: Extract 15×15 patch around maximum
    patch = extract_patch(heatmap, hard_coord, size=15)
    
    # Step 3: Soft-argmax on patch
    offset = soft_argmax(patch)  # Offset from patch center
    
    # Step 4: Combine
    final_coord = hard_coord + offset - patch_size//2
    
    return final_coord
```

---

## 1. THREE MAIN MODELS

### **Model 1: Refiner (RITM_SE_HRNet32)**
- **Architecture**: HRNet-32 with SE-Net attention
- **Parameters**: ~30-40M
- **Input**: 
  - RGB Image (3 × 512 × 256)
  - Hint Heatmaps (68 or 136 channels)
- **Output**: 68 keypoint heatmaps (512 × 256)
- **Role**: Primary keypoint prediction

### **Model 2: Detector (ResNet-18)**
- **Architecture**: ResNet-18 (modified first conv layer)
- **Parameters**: ~11M
- **Input**: Image (3 channels) + Predicted Heatmaps (68) = 71 channels
- **Output**: 68 binary probabilities (error likelihood per keypoint)
- **Role**: Identify erroneous keypoints

### **Model 3: Corrector (DeepLabV3)**
- **Architecture**: DeepLabV3 + ResNet-50 backbone
- **Parameters**: ~40M
- **Input**: Image (3 channels) + Predicted Heatmaps (68) = 71 channels
- **Output**: 68 refined heatmaps (256 × 128)
- **Role**: Generate corrected keypoint positions

---

## 2. HOW EACH MODEL IS TRAINED

### **A. Refiner Training**

**Dataset**:
- Training: 226 images × 68 keypoints
- Validation: 128 images × 68 keypoints
- Augmentation: Rotation ±15°, flip, scale 0.9-1.2×

**Training Process**:
```
For each epoch (max 1000):
  For each batch (size=4):
    1. Random hints: Select 0-68 keypoints
    2. Random iterations (0-3, no gradient):
       - Forward pass
       - Find worst prediction (highest error)
       - Add as hint for next iteration
    3. Final forward pass WITH gradient
    4. Compute loss: BCE + Distance + Angle
    5. Backpropagation
    6. Update weights
  
  Validation:
    - Forward pass on validation set
    - Compute MRE (Mean Radial Error)
    - Save if MRE improved
    - Early stop if no improvement for 50 epochs
```

**Loss Function**:
```
L_total = L_BCE + 0.01×L_distance + 0.01×L_angle

Where:
- L_BCE: Binary cross-entropy on heatmaps
- L_distance: L1 loss on 70 bone segment lengths
- L_angle: Cosine embedding loss on 70 joint angles
```

**Optimizer**: Adam (lr=0.001)
**Training Time**: ~24 hours (GPU)
**Best Model**: Lowest validation MRE in millimeters

---

### **B. Detector Training**

**Dataset Creation**:
```
1. Use TRAINED Refiner to predict on train/val sets
2. For each keypoint in each image:
   - Compare prediction to ground truth
   - If MRE > threshold → label = 1 (needs correction)
   - If MRE ≤ threshold → label = 0 (correct)
3. Result: Binary classification dataset
```

**Training Process**:
```
For each epoch (max 300):
  For each batch (size=128):
    1. Get: image, keypoint_coords, binary_labels
    2. Generate heatmaps from coords
    3. Forward: predictions = model(image, heatmaps)
    4. Compute loss: BCE(predictions, binary_labels)
    5. Backpropagation
    6. Update weights
  
  Validation:
    - Compute accuracy: (predictions > 0.5) == labels
    - Save if accuracy improved
```

**Optimizer**: AdamW (lr=1e-3)
**Training Time**: ~3 hours (GPU)
**Best Model**: Highest validation accuracy
**Performance**: 85% precision, 71% recall

---

### **C. Corrector Training**

**Dataset Creation**:
```
1. Use TRAINED Refiner to predict on train/val sets
2. For each image:
   - Input: Erroneous predictions (keypoint_neg)
   - Target: Ground truth positions (keypoint_pos)
3. Pairs represent: "Given wrong prediction, output correct position"
```

**Training Process**:
```
For each epoch (max 300):
  For each batch (size=4):
    1. Get: image, keypoint_neg, keypoint_pos
    2. Generate input heatmaps from keypoint_neg
    3. Generate target heatmaps from keypoint_pos
    4. Forward: pred_heatmaps = model(image, input_heatmaps)
    5. Compute loss: BCE(pred_heatmaps, target_heatmaps)
    6. Backpropagation
    7. Update weights
  
  Validation:
    - Compute loss on validation set
    - Extract coords: heatmap → hard-argmax
    - Compute MRE: sqrt((pred - gt)²)
    - Save if loss improved
```

**Optimizer**: AdamW (lr=1e-3)
**Training Time**: ~5 hours (GPU)
**Best Model**: Lowest validation loss
**Performance**: 78% improvement rate

---

## 3. HOW MODELS ARE VALIDATED

### **Refiner Validation**:
- **Metric**: Mean Radial Error (MRE) in millimeters
- **Method**: Hard-argmax coordinate extraction from heatmaps
- **Frequency**: Every epoch
- **Decision**: Save if validation MRE decreases

### **Detector Validation**:
- **Metric**: Binary accuracy
- **Method**: (predictions > 0.5) == ground_truth_labels
- **Frequency**: Every epoch
- **Decision**: Save if validation accuracy increases

### **Corrector Validation**:
- **Primary Metric**: Heatmap BCE loss
- **Secondary Metric**: MRE after coordinate extraction
- **Method**: Generate heatmaps, extract coords, compare to GT
- **Frequency**: Every epoch
- **Decision**: Save if validation loss decreases

---

## 4. HOW MODELS COLLABORATE (INFERENCE)

### **Workflow: Iterative Refinement**

```
┌─────────────────────────────────────────────────────────┐
│ INITIALIZATION                                          │
│ - User provides 0-5 hints (optional)                   │
│ - Create hint heatmaps from user input                 │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 1: REFINER (Initial Prediction)                   │
│                                                         │
│ Input:  Image + Hint Heatmaps                          │
│ Output: 68 predicted heatmaps                          │
│ Action: heatmaps → soft-argmax → 68 coordinates       │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ ITERATION LOOP (Repeat 3 times)                        │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │ STEP 2: DETECTOR (Error Identification)        │  │
│  │                                                 │  │
│  │ Input:  Image + Predicted Heatmaps             │  │
│  │ Output: 68 error probabilities                 │  │
│  │ Action: Threshold at 0.5 → binary error mask   │  │
│  └──────────────────┬──────────────────────────────┘  │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Check: Any errors detected?                     │  │
│  └─────┬─────────────────────────────┬─────────────┘  │
│        │ YES                          │ NO             │
│        ▼                              ▼                │
│  ┌────────────────────────┐    ┌──────────────┐      │
│  │ STEP 3: CORRECTOR      │    │ BREAK LOOP   │      │
│  │ (Error Correction)     │    └──────────────┘      │
│  │                        │                           │
│  │ Input:  Image +        │                           │
│  │         Heatmaps +     │                           │
│  │         Error Mask     │                           │
│  │ Output: Refined        │                           │
│  │         Heatmaps       │                           │
│  └──────────┬─────────────┘                           │
│             │                                          │
│             ▼                                          │
│  ┌─────────────────────────────────────────────────┐  │
│  │ STEP 4: UPDATE (Selective Refinement)          │  │
│  │                                                 │  │
│  │ For keypoints flagged as errors:                │  │
│  │   - Replace with corrected heatmaps            │  │
│  │ For keypoints marked as correct:               │  │
│  │   - Keep original predictions                  │  │
│  │                                                 │  │
│  │ Update previous heatmap for next iteration     │  │
│  └──────────────────┬──────────────────────────────┘  │
│                     │                                   │
│                     └──────► Back to STEP 2 (DETECTOR) │
│                                                         │
└─────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ FINAL OUTPUT                                            │
│ - 68 refined keypoint coordinates                      │
│ - Confidence: High (after 3 refinement iterations)     │
└─────────────────────────────────────────────────────────┘
```

### **Key Collaboration Features**:

1. **Selective Refinement**:
   - Only keypoints flagged by Detector are corrected
   - Correct predictions remain untouched
   - Prevents unnecessary changes

2. **Iterative Improvement**:
   - Each iteration fixes some errors
   - Detector re-evaluates after each correction
   - Typically converges in 2-3 iterations

3. **Previous Heatmap Feedback**:
   - Refiner receives previous predictions
   - Input channels: 68 (current hints) + 68 (previous predictions) = 136
   - Enables context-aware refinement

4. **User Integration**:
   - User hints treated as ground truth
   - Model predictions + user hints combined
   - Interactive refinement possible

---

## 5. SEQUENTIAL TRAINING DEPENDENCY

### **Why Sequential Training?**

```
Stage 1: Train Refiner
    ↓
    Produces predictions on train/val sets
    ↓
Stage 2: Train Detector
    Uses Refiner predictions to create error labels
    ↓
    Can identify which keypoints need correction
    ↓
Stage 3: Train Corrector
    Uses Refiner predictions as input
    Learns to refine erroneous predictions
    ↓
    Complete system ready
```

**Dependencies**:
- Detector needs Refiner's predictions to create training labels
- Corrector needs Refiner's predictions to learn error patterns
- Cannot train in parallel—must be sequential

**Benefits**:
- Each model has focused, clear objective
- Easier to debug and improve individual components
- More stable training than end-to-end
- Modular: can swap/upgrade components independently

---

## 6. COMPLETE TRAINING TIMELINE

```
Total Training Time: ~30-32 hours (single GPU)

Hour 0-24:   Refiner Training
             - 1000 epochs max
             - Typically stops at 200-400 epochs (early stopping)
             - Save: save/AASCE_interactive_keypoint_estimation/model.pth

Hour 24-27:  Detector Training
             - Generate training data from Refiner predictions
             - Train for 300 epochs
             - Save: save_suggestion/AASCE_suggestModel.pth

Hour 27-32:  Corrector Training
             - Generate training data from Refiner predictions
             - Train for 300 epochs
             - Save: save_refine/AASCE_refineModel.pth
```

---

## 7. MODEL PERFORMANCE SUMMARY

| Model | Input Size | Output Size | Params | Training Time | Key Metric |
|-------|-----------|-------------|---------|---------------|------------|
| Refiner | 3×512×256 + 136 | 68×512×256 | ~40M | 24h | MRE: 3.2mm (0 hints) |
| Detector | 71×512×256 | 68 binary | ~11M | 3h | Accuracy: 85% |
| Corrector | 71×512×256 | 68×256×128 | ~40M | 5h | Improvement: 78% |

---

## 8. KEY COLLABORATION INSIGHTS

1. **Automatic Error Correction**:
   - Reduces user effort by 60-70%
   - User provides 0-5 hints instead of marking all 68 points
   - System automatically identifies and fixes remaining errors

2. **Progressive Refinement**:
   - Iteration 0: 3.2mm MRE
   - Iteration 1: 2.1mm MRE (58% errors fixed)
   - Iteration 2: 1.6mm MRE (78% errors fixed)
   - Iteration 3: 1.4mm MRE (85% errors fixed)

3. **Robustness to Errors**:
   - Detector false positives (15%): Minor unnecessary corrections
   - Detector false negatives (29%): Caught in next iteration
   - Corrector failures (22%): Usually small degradations (~0.3mm)

4. **Clinical Workflow**:
   - Traditional: 15 minutes manual annotation
   - KeyBot (0 hints): 5 seconds automatic
   - KeyBot (3 hints): 3 minutes with review
   - **Result: 80% time reduction**

---

## SUMMARY

**Components**: 3 neural network models
**Training**: Sequential (Refiner → Detector → Corrector)
**Collaboration**: Iterative refinement with selective correction
**Total Time**: ~30 hours training, ~2 seconds inference
**Performance**: 3.2mm → 1.4mm MRE with 3 user hints

