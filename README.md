# SLIC Superpixels & Grad-CAM

This project explores two important areas of computer vision:

1. Image segmentation using SLIC (Simple Linear Iterative Clustering) superpixels
2. Neural network interpretability using gradient-based visualization methods

The project combines algorithm implementation with visual analysis to better understand both low-level image segmentation and high-level model attention.

## Project Overview

The project contains two main components:

### Part 1: SLIC Superpixel Segmentation

SLIC groups nearby pixels into perceptually meaningful regions by balancing:

- Color similarity
- Spatial proximity

Instead of clustering across the entire image, the implementation restricts centroid searches to local neighborhoods to improve computational efficiency.

### Part 2: Visual Attention in Deep Neural Networks

The project also explores several gradient-based interpretation methods:

- Grad-CAM
- Vanilla Backpropagation
- Guided Grad-CAM

These methods were used to visualize which parts of an image contributed most strongly to a model's prediction.

## SLIC Implementation

I implemented the SLIC algorithm from scratch.

The algorithm represents each pixel using both color and spatial information and assigns pixels to nearby cluster centers based on a combined distance metric.

To improve efficiency, cluster-center searches were restricted to local `2S × 2S` regions rather than evaluating all centroids globally.

## Superpixel Granularity

Different values of K were tested to study how the number of superpixels affects segmentation quality.

Experiments included:

- K = 50
- K = 200
- K = 500

Lower K values produced larger superpixels that captured broad image regions.

Higher K values produced finer superpixels that better followed detailed object boundaries.

## Comparison with Scikit-image

The custom SLIC implementation was compared with the optimized implementation from `scikit-image`.

The comparison focused on:

- Boundary adherence
- Cluster regularity
- Stability
- Computational performance

The custom implementation successfully produced meaningful superpixels, while the library implementation produced more regular and stable regions due to additional optimization and connectivity post-processing.

## Model Interpretability

The second part of the project investigates how deep neural networks focus on different image regions.

### Grad-CAM

Grad-CAM uses gradients flowing into a convolutional layer to generate a localization heatmap that highlights regions important to the model's prediction.

### Vanilla Backpropagation

Vanilla backpropagation visualizes gradients with respect to the input image, showing which pixels most strongly influence the output.

### Guided Grad-CAM

Guided Grad-CAM combines coarse Grad-CAM localization with higher-resolution guided backpropagation information.

## Layer-wise Analysis

Attention maps were generated at different network layers to study how representations change with network depth.

Earlier layers generally capture lower-level visual patterns, while deeper layers tend to focus on more semantically meaningful image regions.

## Robustness Experiments

The project also tested attention-map stability under image transformations such as rotations.

This allowed the visualization methods to be evaluated not only qualitatively, but also in terms of how consistent their localization behavior remained under altered inputs.

## Technologies

- Python
- NumPy
- PyTorch
- Scikit-image
- Matplotlib
- SLIC
- Grad-CAM
- Guided Grad-CAM
- Computer Vision
- Explainable AI

## Repository Structure

```text
slic-superpixels-gradcam/
├── README.md
├── notebooks/
│   └── slic_gradcam.ipynb
├── src/
│   ├── slic.py
│   └── gradcam.py
├── results/
│   ├── slic_comparisons/
│   └── gradcam_visualizations/
├── docs/
│   └── report.pdf
└── .gitignore
