# GAN-Based Noise Generation and Classification

## Project Overview

This project explores Generative Adversarial Networks (GANs) to enhance a model's ability to differentiate between class 1 samples and noise. It uses a dataset containing only samples labeled as class 1, applies GANs to generate realistic noise samples, and trains a classifier model to distinguish between the generated noise and original class 1 data.

## Table of Contents

- [GAN-Based Noise Generation and Classification](#gan-based-noise-generation-and-classification)
  - [Project Overview](#project-overview)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Implementation Details](#implementation-details)
    - [Data Preprocessing](#data-preprocessing)
    - [GAN Architecture](#gan-architecture)
      - [Generator Network](#generator-network)
      - [Discriminator Network](#discriminator-network)
    - [Noise Generation](#noise-generation)
    - [Classification Model](#classification-model)
  - [Results and Evaluation](#results-and-evaluation)
  - [How to Use](#how-to-use)
  - [Dependencies](#dependencies)

## Project Structure

The repository contains the following key files:

- `main_final.ipynb`: Jupyter notebook containing the complete implementation
- `Dataset_timeseries.csv`: Original dataset with class 1 samples
- `Generator.h5`: Saved GAN generator model
- `Noise Classifier.h5`: Saved binary classifier model
- `final_dataset.csv`: Combined dataset with both real and generated samples
- `classification_report.pdf`: Performance metrics of the classifier
- `Confusion Matrix.jpg`: Visual representation of model predictions
- `KDE Comparision.jpg`: Kernel Density Estimation comparing real vs generated data
- `GAN Figures.jpg`: Visualization of GAN architecture and results
- `Report.pdf`: Detailed project report

## Implementation Details

### Data Preprocessing

The implementation begins by loading a time series dataset that contains samples labeled as class 1. The preprocessing steps include:

- Loading the dataset and extracting class 1 samples with their 'SOPAS' feature
- Using Isolation Forest to detect outliers with a contamination rate of 5%
- Normalizing the data to the range (-1, 1) using MinMaxScaler for optimal GAN training
- Identifying and separating outliers to help train the GAN on anomalous patterns

### GAN Architecture

The GAN implementation consists of two primary components:

#### Generator Network
- Input: Random noise vector of dimension 200
- Architecture: Four dense layers with increasing neuron counts (256, 512, 1024, 2048)
- Each layer uses LeakyReLU activation (alpha=0.01) and batch normalization
- Output layer uses tanh activation to generate values in the normalized range (-1, 1)

#### Discriminator Network
- Input: One-dimensional data point
- Architecture: Four dense layers with increasing neuron counts (256, 512, 1024, 2048)
- Each layer uses LeakyReLU activation (alpha=0.01) and dropout (rate=0.4) for regularization
- Output layer uses sigmoid activation for binary classification (real vs fake)

The GAN is optimized using the Adam optimizer with a learning rate of 0.0002 and beta_1=0.5, which are commonly used hyperparameters for GANs. The training process uses a custom training function that implements alternating updates of the discriminator and generator.

### Noise Generation

After training the GAN for 20,000 epochs, the generator is saved and then used to create synthetic noise samples:

- Random noise vectors are generated from a normal distribution
- The trained generator transforms these vectors into synthetic data points
- Generated samples are inverse-transformed to match the scale of the original data
- Statistical validation confirms the generated noise has similar properties to real outliers
- A Kernel Density Estimation (KDE) plot visually compares the distributions

### Classification Model

The final stage involves building a binary classifier to distinguish between normal data and generated noise:

- A combined dataset is created with original class 1 samples (label 1) and generated noise (label 0)
- Data is normalized and split into training (80%) and testing (20%) sets
- A simple neural network with one hidden layer (32 neurons) is implemented
- The model uses binary cross-entropy loss and Adam optimizer
- Training runs for 50 epochs with a batch size of 64 and 20% validation split

## Results and Evaluation

The classification model's performance is evaluated using standard metrics:

- Confusion matrix visualization shows the distribution of correct and incorrect predictions
- Classification report includes precision, recall, and F1-score for each class
- The model achieves high accuracy in distinguishing between normal data and generated noise
- Results are saved to a PDF report for documentation

## How to Use

To utilize this project's models:

1. Load the pre-trained generator to create noise samples:
   - Load the model using `tf.keras.models.load_model('Generator.h5')`
   - Generate noise vectors and pass them through the model
   - Inverse transform the output to match the original data scale

2. Use the classifier to distinguish between normal data and noise:
   - Load the model using `tf.keras.models.load_model('Noise Classifier.h5')`
   - Normalize input data before prediction
   - Apply threshold of 0.5 to convert probabilities to binary predictions

## Dependencies

- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn