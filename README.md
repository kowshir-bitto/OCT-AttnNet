# OCT-AttnNet

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

**OCT-AttnNet** is a PyTorch-based deep learning project for **Optical Coherence Tomography (OCT) image classification**. It combines transfer learning with attention mechanisms and explainable AI (XAI) methods to support robust and interpretable image classification experiments.

The repository includes baseline CNN architectures, an attention-enhanced **InceptionV3 + BAM + ECA** model, preprocessing utilities, class-balancing augmentation, model evaluation, and visual explanation tools.

## Key Features

- PyTorch-based OCT image classification pipeline
- Transfer learning with pretrained CNN architectures
- Baseline experiments with:
  - VGG16
  - VGG19
  - ResNet50
  - InceptionV3
- Attention-enhanced InceptionV3 using:
  - **BAM** — Bottleneck Attention Module
  - **ECA** — Efficient Channel Attention
- CLAHE-based image enhancement
- Image resizing and train/test splitting
- Data augmentation and minority-class oversampling
- Mixed-precision training support
- Classification report and confusion matrix
- Per-class TP, TN, FP, FN, sensitivity/recall, specificity, FPR, and FNR
- Explainable AI with:
  - Integrated Gradients
  - Grad-CAM
  - Grad-CAM++

## Proposed Architecture

```text
OCT Image
   │
   ▼
Preprocessing / Enhancement
   │
   ├── CLAHE
   ├── Resize
   └── Data Balancing / Augmentation
   │
   ▼
Pretrained InceptionV3
   │
   ▼
Deep Feature Extraction
   │
   ▼
BAM Attention
(Channel + Spatial Attention)
   │
   ▼
ECA Attention
(Efficient Channel Attention)
   │
   ▼
Classification Layer
   │
   ▼
Predicted OCT Class
   │
   ▼
Explainability
(Integrated Gradients / Grad-CAM / Grad-CAM++)
```

## Attention Modules

### Bottleneck Attention Module (BAM)

The implemented BAM combines **channel attention** and **spatial attention**. Channel attention uses global average pooling and convolutional layers, while spatial attention uses convolutional operations with dilation to learn spatially important regions.

### Efficient Channel Attention (ECA)

ECA applies global average pooling followed by lightweight one-dimensional convolution to model channel-wise dependencies without a large parameter overhead.

In the attention-enhanced model, BAM and ECA are attached to the final InceptionV3 mixed block.

## Repository Structure

```text
OCT-AttnNet/
├── CLAHE.py
├── Resize and Split.py
├── Oversample.py
│
├── VGG16.py
├── VGG19.py
├── ResNet50.py
├── InceptionV3.py
├── InceptionV3+BAM+ECA.py
│
├── attention_modules.py
├── data_handler.py
├── model_builder.py
├── train.py
├── evaluate.py
├── xai_explainer.py
│
└── LICENSE
```

## Preprocessing Pipeline

### 1. CLAHE Enhancement

`CLAHE.py` applies **Contrast Limited Adaptive Histogram Equalization (CLAHE)** in LAB color space to improve local image contrast.

The current script uses:

```python
cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)
```

### 2. Resize and Split

Use:

```text
Resize and Split.py
```

to prepare resized images and organize the dataset into training/testing subsets.

### 3. Class Balancing

`Oversample.py` balances minority classes through image augmentation. The augmentation pipeline includes operations such as:

- Horizontal flipping
- Vertical flipping
- Brightness adjustment
- Contrast adjustment
- Saturation adjustment
- Hue adjustment

## Models

### Baseline Models

The repository provides scripts for:

```text
VGG16
VGG19
ResNet50
InceptionV3
```

These models can be used as baseline architectures for comparison with the proposed attention-enhanced approach.

### OCT-AttnNet / Attention Model

The main attention-based experiment is implemented in:

```text
InceptionV3+BAM+ECA.py
```

The model uses an ImageNet-pretrained InceptionV3 backbone and adds BAM and ECA attention modules to the `Mixed_7c` stage before classification.

## Training

The training utilities are provided in:

```text
train.py
```

The training loop supports:

- Cross-entropy-based classification
- InceptionV3 auxiliary logits
- Automatic mixed precision
- Gradient scaling
- Training loss tracking
- Validation loss tracking
- Training accuracy tracking
- Validation accuracy tracking

## Evaluation

`evaluate.py` calculates overall model performance and detailed per-class statistics.

Evaluation outputs include:

- Test loss
- Test accuracy
- Classification report
- Confusion matrix
- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)
- True Positive Rate / Recall
- True Negative Rate / Specificity
- False Positive Rate
- False Negative Rate
- Per-class accuracy

## Explainable AI

The project contains an XAI module in:

```text
xai_explainer.py
```

Supported explanation methods include:

### Integrated Gradients

Highlights image regions that contribute to the predicted class by integrating gradients between a baseline and the input image.

### Grad-CAM

Produces activation heatmaps from deep convolutional features to visualize regions influencing a prediction.

### Grad-CAM++

Provides a refined class-discriminative localization map and can provide improved visualization when multiple relevant regions are present.

## Installation

Clone the repository:

```bash
git clone https://github.com/kowshir-bitto/OCT-AttnNet.git
cd OCT-AttnNet
```

Install the primary dependencies:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn pillow opencv-python captum grad-cam
```

## Dataset Organization

The data loader expects images to be organized by class folders:

```text
dataset/
├── Class_1/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── Class_2/
│   ├── image_001.jpg
│   └── ...
└── Class_N/
    └── ...
```

Class labels are generated automatically from the names of the class directories.

The preprocessing scripts currently reference dataset folders outside the repository, for example:

```text
../Datasets/0. Raw/
../Datasets/1. Enhanched/
../Datasets/2. Split/
../Datasets/3. Balanced/
```

Update these paths according to your local dataset location.

## Running the Project

A typical workflow is:

```text
Raw OCT Dataset
      │
      ▼
CLAHE.py
      │
      ▼
Resize and Split.py
      │
      ▼
Oversample.py
      │
      ▼
Baseline / Attention Model
      │
      ▼
Evaluation
      │
      ▼
XAI Visualization
```

For the attention-enhanced experiment, use:

```bash
python "InceptionV3+BAM+ECA.py"
```

> **Note:** Some current scripts reference a `config.py` file and modules under a `Functions` package. If you are running the repository exactly as currently uploaded, you may need to add the corresponding configuration file or adjust those imports to match the root-level file structure.

## Requirements

Main libraries used by the project include:

```text
Python
PyTorch
torchvision
NumPy
Matplotlib
Seaborn
scikit-learn
Pillow
OpenCV
Captum
pytorch-grad-cam
```

A CUDA-capable GPU is recommended for model training.

## Research Use

This repository is designed for experiments involving:

- OCT image classification
- Transfer learning
- Attention mechanisms
- CNN architecture comparison
- Medical image preprocessing
- Explainable artificial intelligence
- Model interpretability

## License

This repository is currently distributed under the **MIT**.

See the [LICENSE](LICENSE) file for details.

## Author

**Abu Kowshir Bitto**

- GitHub: [@kowshir-bitto](https://github.com/kowshir-bitto)
- Website: [kowshirbitto.me](http://kowshirbitto.me/)
- Google Scholar: [Abu Kowshir Bitto](https://scholar.google.com/citations?hl=en&user=AO0dWsgAAAAJ&view_op=list_works&gmla=AJ1KiT30Ms5pY2DUl6pfWl4cwjlBOwygW_3wawpWiD_769YBbLX8_0rqv4MiIf05GjDe6xY81ApN7Gy1DfwYJCZu)
