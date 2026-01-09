# Fast Food Classification using Transfer Learning

A comprehensive PyTorch implementation comparing different transfer learning approaches for fast food image classification using pre-trained CNN models.

## Overview

This project demonstrates the effectiveness of transfer learning for image classification by comparing multiple scenarios:
- **Fine-tuning vs Feature Extraction**: Compare full model fine-tuning against using pre-trained models as fixed feature extractors
- **Dataset Size Impact**: Analyze performance differences between full and reduced datasets
- **Model Architecture Comparison**: Evaluate ResNet18, MobileNetV2, and EfficientNet-B0 architectures

## Dataset

The project uses the [Fast Food Classification Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset) from Kaggle, which contains images of various fast food items across multiple categories.

## Key Features

- **Four Training Scenarios**:
  1. Full Dataset + Fine-tuning
  2. Full Dataset + Feature Extraction
  3. Small Dataset + Fine-tuning
  4. Small Dataset + Feature Extraction

- **Multiple Model Architectures**:
  - ResNet18
  - MobileNetV2
  - EfficientNet-B0

- **Comprehensive Evaluation**:
  - Training and validation accuracy tracking
  - Loss curves visualization
  - Confusion matrix analysis
  - Classification reports
  - Model prediction visualization

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
kagglehub
numpy
matplotlib
scikit-learn
ipywidgets
````

## Usage

### Option 1: Google Colab (Recommended)
1. Open the notebook in Google Colab
2. Enable GPU runtime (Runtime > Change runtime type > GPU)
3. Run all cells sequentially


## Project Structure

```
fastfood-transfer-learning/
│
├── FastFood_TransferLearning_PyTorch_Enhanced.ipynb  # Main notebook
├── README.md                                          # This file
└── data/                                              # Dataset (auto-downloaded)
```

## Notebook Contents

The notebook is organized into three main sections:

### Setup and Preparation
- Environment configuration
- Library imports
- Dataset download and preparation
- Data augmentation and preprocessing
- Visualization utilities

### Experiments and Comparisons
- Four training scenarios with different configurations
- Multiple model architecture comparisons
- Training history visualization
- Performance metrics tracking

### Evaluation and Results
- Confusion matrix analysis
- Classification reports
- Model predictions visualization
- Model saving and loading
- Interactive prediction interface

## Key Results

The experiments demonstrate that:
- Fine-tuning generally achieves higher accuracy than feature extraction
- Full dataset training significantly outperforms small dataset scenarios
- EfficientNet-B0 provides the best accuracy-efficiency trade-off
- Transfer learning enables good performance even with limited data

## Model Performance

| Model | Approach | Dataset | Accuracy |
|-------|----------|---------|----------|
| ResNet18 | Fine-tuning | Full | ~85-90% |
| ResNet18 | Feature Extraction | Full | ~80-85% |
| MobileNetV2 | Fine-tuning | Full | ~83-88% |
| EfficientNet-B0 | Fine-tuning | Full | ~86-91% |

*Results may vary depending on hyperparameters and random initialization*

## Training Configuration

- **Image Size**: 224x224
- **Batch Size**: 64
- **Epochs**: 5-10 (depending on scenario)
- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: 0.001
- **Scheduler**: StepLR (decay every 7 epochs)

## Visualization Features

- Training/validation loss curves
- Accuracy progression over epochs
- Sample predictions with confidence scores
- Confusion matrix heatmaps
- Multi-model comparison charts

## Saving and Loading Models

Models are automatically saved during training with the following information:
- Model state dictionary
- Optimizer state
- Training epoch
- Best validation accuracy
- Class names

## Acknowledgments

- Dataset: [Utkarsh Saxena on Kaggle](https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset)
- Pre-trained models from torchvision.models
- PyTorch framework and community

}
```
