# Multi-Class Chest X-Ray Classification with EfficientNet-B0

This repository contains the code and documentation for the paper "Multi-Class Chest X-Ray Classification with EfficientNet-B0". The goal of this project is to classify chest X-ray (CXR) images into four categories: COVID-19, pneumonia, tuberculosis (TB), and healthy lungs using the EfficientNet-B0 model. Click here: ([mendeley_model.html](https://noahba65.github.io/EfficientNetB0_x_ray_classification/mendeley_model.html)) to view the JupyterLabs Notebook code.

## Overview

Deep Learning (DL) has been successfully applied to classify conditions affecting the chest using X-rays (CXR). While most studies have focused on binary classifications, this project extends these efforts by employing the pre-trained EfficientNet-B0 model to classify a multi-class dataset. The dataset is constructed from the Mendeley CXR dataset and the Mendeley TB dataset. The model achieved an accuracy of 99.57% and an F1 score of 0.9959 on an unseen test sample.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Experimental Design](#experimental-design)
- [Results](#results)
- [Discussion](#discussion)
- [References](#references)

## Installation

To run the code in this repository, you need to have Python and PyTorch installed. Follow these steps to set up your environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/CXR-Classification.git
    cd CXR-Classification
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The data used in this study were sourced from two distinct datasets available on Mendeley Data:

1. **[Mendeley CXR dataset](https://data.mendeley.com/datasets/8gf9vpkhgy/1)**: Includes images of COVID-19, pneumonia, and healthy lungs (Danilov, 2022).
2. **[Mendeley TB dataset](https://data.mendeley.com/datasets/8j2g3csprk/2)**: Includes images of tuberculosis (TB) (Kieran, 2024).

To ensure approximate class balance, a subset of TB images was carefully selected. The final dataset used in this study contains the following number of images per class:

- COVID-19: 1,627 images
- Normal: 1,803 images
- Pneumonia: 1,801 images
- TB: 1,795 images

## Model Architecture

EfficientNet-B0 is known for its high accuracy and efficiency in image classification tasks. The B0 model serves as the foundational model in the B-series, featuring mobile inverted bottleneck (MBConv) layers. The detailed architecture of the B0 model includes seven MBConv stages and operates at 0.39 GFLOPS (Tan et al. 2019).

## Experimental Design

### Data Preprocessing and Augmentation

- Images were resized to 224x224 pixels and converted to PyTorch tensors.
- Data augmentation techniques, such as axis flipping and random rotations, were tested but did not yield significant benefits.

### Model Metrics

The selected metrics include Accuracy, Precision, Recall, and F1 scores, all of which are Macro-averaged metrics.

### Model Implementation

- The dataset was split into training (80%), validation (10%), and test (10%) sets.
- The cross entropy loss function and Adam optimizer with a learning rate of 0.001 were used.
- A Reduce on Plateau learning rate scheduler was applied.
- The model was trained for 30 epochs with a batch size of 32.

## Results

The model achieved the following performance metrics on the test set:

- **Accuracy**: 99.57%
- **Precision**: 99.60%
- **Recall**: 99.57%
- **F1 Score**: 99.59%

These results demonstrate that the EfficientNet-B0 model outperformed existing models in the literature for four-class CXR classification.

## Discussion

The EfficientNet-B0 model, despite its simplicity and lightweight design, shows promising results for multi-class CXR classification. Future work could involve training and testing the model on more diverse datasets and exploring its application to other medical imaging tasks.

## References

- Kiran, Saira; Jabeen, Dr Ishrat (2024), “Dataset of Tuberculosis Chest X-rays Images”, Mendeley Data, V2, doi: 10.17632/8j2g3csprk.2
- Danilov, Viacheslav; Proutski, Alex; Kirpich, Alexander; Litmanovich, Diana; Gankin, Yuriy (2022), “Chest X-ray dataset for lung segmentation”, Mendeley Data, V1, doi: 10.17632/8gf9vpkhgy.1
- Tan, Mingxing and Le, Quoc. Efficientnet: Rethinking model scaling for
convolutional neural networks. In International conference on machine
learning, pages 6105–6114. PMLR, 2019.

## Contact

For any questions or comments, please contact Noah Anderson at noahanderson6556@gmail.com.

---

Feel free to modify this README file as needed to better suit your project's structure and details.
