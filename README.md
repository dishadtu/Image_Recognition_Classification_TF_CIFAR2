# Image_Recognition_Classification_TF_CIFAR2

This project implements an image recognition classification model using TensorFlow for the TF_CIFAR2 dataset. The TF_CIFAR2 dataset consists of 60,000 32x32 color images in 10 different classes. This repository includes code for data preprocessing, model building, training, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Saving the Model](#saving-the-model)
  

## Introduction

This project demonstrates how to create a convolutional neural network (CNN) using TensorFlow to classify images from the CIFAR-10 dataset. The model is trained to recognize objects in images and can predict labels for unseen images.

## Installation

To run this project, you need to have Python and TensorFlow installed. `

## Dataset

The CIFAR-10 dataset is included in TensorFlow and can be easily loaded using the `tf.keras.datasets` module. The dataset consists of 60,000 images divided into 10 classes:

* **Airplane**
* **Automobile**
* **Bird**
* **Cat**
* **Deer**
* **Dog**
* **Frog**
* **Horse**
* **Ship**
* **Truck**

### Dataset Structure

* **Training Set:** 50,000 images
* **Test Set:** 10,000 images

Each image is of size 32x32 pixels and contains three color channels (RGB).

## Model Architecture

The model is built using the Keras Functional API and consists of the following layers:

1. **Input Layer:** Accepts input images.
2. **Convolutional Layers:** Extract features from images using various filters.
3. **Batch Normalization Layers:** Normalize activations to improve convergence.
4. **Max Pooling Layers:** Downsample feature maps to reduce dimensionality.
5. **Flatten Layer:** Converts the 3D output to 1D for the dense layer.
6. **Dense Hidden Layer:** Fully connected layer for high-level reasoning.
7. **Output Layer:** Uses softmax activation to output class probabilities.

## Training

The model is trained using the Adam optimizer and sparse categorical cross-entropy loss function. Training consists of two phases:

1. **Initial Training:** 10 epochs on the original dataset.
2. **Data Augmentation:** Further training for 5 epochs using data augmentation techniques, including:
   * Width shifts
   * Height shifts
   * Horizontal flips

## Results

After training, the model's accuracy on the training and validation datasets is plotted to visualize performance. You can visualize the accuracy using Matplotlib.

## Saving the Model

The trained model is saved in multiple formats for future use:

* **HDF5 format:** `imagerecognition_tf_cifar.h5`
* **TensorFlow SavedModel format:** `saved_model/my_model`
* **Model weights:** `gfgModelWeights`

