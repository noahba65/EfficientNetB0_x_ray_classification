# Multi-Class Chest X-Ray Classification with EfficientNet-B0

This repository contains the code and documentation for the paper "Multi-Class Chest X-Ray Classification with EfficientNet-B0". The goal of this project is to classify chest X-ray (CXR) images into four categories: COVID-19, pneumonia, tuberculosis (TB), and healthy lungs using the EfficientNet-B0 model.

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

1. **Mendeley CXR dataset**: Includes images of COVID-19, pneumonia, and healthy lungs.
2. **Mendeley TB dataset**: Includes images of tuberculosis (TB).

To ensure approximate class balance, a subset of TB images was carefully selected. The final dataset used in this study contains the following number of images per class:

- COVID-19: 1,627 images
- Normal: 1,803 images
- Pneumonia: 1,801 images
- TB: 1,795 images

## Model Architecture

EfficientNet-B0 is known for its high accuracy and efficiency in image classification tasks. The B0 model serves as the foundational model in the B-series, featuring mobile inverted bottleneck (MBConv) layers. The detailed architecture of the B0 model includes seven MBConv stages and operates at 0.39 GFLOPS.

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

- Nasiri et al., 2022
- Karakanis et al., 2021
- Kaur et al., 2023
- Venkataramana et al., 2022
- Kansal et al., 2023
- Bashar et al., 2021
- Kim et al., 2022
- Bhandarani et al., 2022
- Ahmed et al., 2023
- Tan et al., 2019 (EfficientNet)
- Sandler et al., 2018 (MobileNetV2)
- PyTorch Documentation
- Mendeley Data

## Contact

For any questions or comments, please contact Noah Anderson at nba6128@truman.edu.

---

Feel free to modify this README file as needed to better suit your project's structure and details.
