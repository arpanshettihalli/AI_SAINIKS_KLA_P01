# AI-Based Restoration of Degraded Images

## Overview

This project presents a deep learning-based solution for restoring degraded images affected by noise and resolution degradation.

The proposed approach uses a **12-block residual convolutional neural network (CNN)** with an internal bicubic upsampling baseline. The model takes degraded **128×128 noisy low-resolution images** and produces restored **256×256 images**.

The system is designed for AI-based restoration of degraded images, with the goal of recovering structural details while suppressing degradation and noise.

---

## Proposed Approach

The restoration network uses residual learning over a bicubic baseline.

### Pipeline

```text
Noisy LR Image
   128×128
      │
      ▼
Bicubic ×2 Upsampling
      │
      ▼
Feature Extraction
      │
      ▼
12 Residual Blocks
      │
      ▼
Predicted Residual
      │
      ▼
Bicubic Image + Residual
      │
      ▼
Restored Image
   256×256
```

The bicubic interpolation is performed **inside the model**. The raw noisy 128×128 input is passed directly to the network during inference.

---

## Model Architecture

The final submitted model is a **12-block residual CNN**.

Main components:

- Bicubic ×2 upsampling
- Convolutional feature extraction
- 12 residual refinement blocks
- Residual correction prediction
- Addition of the predicted residual to the bicubic baseline

The residual formulation allows the network to focus on learning the missing details and corrections rather than reconstructing the complete image from scratch.

---

## Loss Function

The model uses a combined loss designed to improve both pixel accuracy and structural detail preservation.

### Components

**MSE Loss**

Provides pixel-level reconstruction accuracy.

**SSIM Loss**

Encourages preservation of structural similarity between the restored image and the ground truth.

**Laplacian / Gradient Loss**

Encourages preservation of edges and high-frequency image information.

The combined objective is:

```text
Total Loss =
MSE Loss
+ α × SSIM Loss
+ β × Gradient/Laplacian Loss
```

---

## Dataset

### Training / Validation

The model is trained using paired low-resolution and ground-truth images.

```text
Input : 128×128
Target: 256×256
```

The degradation pipeline includes degraded/noisy low-resolution images corresponding to the restoration task.

### Test Data

The competition test set contains **400 noisy 128×128 `.npy` images**.

The submitted inference script processes these images and generates 256×256 restored `.npy` outputs.

---

## Validation Results

The final validation performance is:

| Metric | Result |
|---|---:|
| Mean PSNR | **27.7025 dB** |
| Mean SSIM | **0.7481** |

---

## Performance by Difficulty

The validation set was additionally analyzed according to image difficulty.

| Difficulty | Bicubic PSNR | CNN PSNR | PSNR Improvement | Bicubic SSIM | CNN SSIM | SSIM Improvement |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 26.5454 | **30.7178** | **+4.1723** | 0.6826 | **0.8362** | **+0.1536** |
| Medium | 22.3111 | **27.4591** | **+5.1480** | 0.4979 | **0.7457** | **+0.2478** |
| Hard | 19.5879 | **24.9444** | **+5.3565** | 0.3932 | **0.6628** | **+0.2696** |

The model improves over the bicubic baseline across all three difficulty groups.

---

## Deployment Performance

The final standalone inference script was tested on the complete set of 400 test images.

| Parameter | Result |
|---|---:|
| Number of test images | 400 |
| Input resolution | 128×128 |
| Output resolution | 256×256 |
| Model size | **10.3 MB** |
| Total inference time | **150.9 seconds** |
| Average inference time | **~0.38 seconds/image** |
| Output datatype | `float32` |
| Output format | `.npy` |

The measured inference time was obtained using the submitted `run.py` on the local benchmark environment.

---

## Inference

The repository contains a standalone Python inference script.

The evaluator can provide:

1. A directory containing the test `.npy` images
2. A directory where restored outputs should be written

Run:

```bash
python run.py <test_images_directory> <output_directory>
```

### Example

```bash
python run.py "./NoisyLR" "./outputs"
```

or:

```bash
python run.py "D:/Dataset/Test" "D:/Dataset/Restored"
```

No modification of `run.py` is required.

---

## Input and Output

### Input

Each input file:

```text
Shape: (128, 128)
Format: .npy
```

The noisy LR input is passed directly to the model without clipping.

### Output

Each generated file:

```text
Shape: (256, 256)
Datatype: float32
Range: [0, 1]
Format: .npy
```

The final model output is clipped to the valid `[0, 1]` range before being saved.

---

## Repository Structure

```text
team_name/

├── run.py
├── requirements.txt
├── README.md
├── training_notebook_final_12block_laplacian.ipynb
│
└── models/
    └── best_model.keras
```

---

## Model

The trained model is stored at:

```text
models/best_model.keras
```

The inference script automatically locates the model relative to `run.py`.

No model path needs to be manually specified when running the inference script.

---

## Training

The complete training workflow is provided in:

```text
training_notebook_final_12block_laplacian.ipynb
```

The notebook contains:

- Dataset loading
- Data preprocessing
- Data augmentation
- Model construction
- Residual blocks
- Loss function implementation
- Model training
- Validation evaluation
- PSNR and SSIM evaluation
- Image comparisons
- Model saving

---

## Technology Stack

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Reproducibility

The repository provides:

- Standalone inference script
- Trained model
- Training notebook
- Complete environment specification

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## References

1. Wang et al., *Image Quality Assessment: From Error Visibility to Structural Similarity*, IEEE Transactions on Image Processing.
2. Dong et al., *Learning a Deep Convolutional Network for Image Super-Resolution*, ECCV.
3. TensorFlow / Keras documentation.
