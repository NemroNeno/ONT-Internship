# Time Series Classification with Attention Mechanisms

## Project Overview
This project implements advanced deep learning models with attention mechanisms to classify univariate time series data into four distinct classes. The goal was to achieve at least 95% accuracy, which was successfully exceeded with our final model achieving over 99% accuracy.

## Objective
The goal of this task was to develop a classification model that can accurately classify a labeled univariate time series dataset into one of four classes with a target accuracy of 95% or higher, incorporating attention mechanisms to enhance model performance.

## Dataset
The project uses a labeled univariate time series dataset where each data point consists of a time series and its corresponding class label (one of four possible classes). Data preprocessing steps included handling missing values, outliers, normalization, and feature engineering to optimize model performance.

## Project Structure
- `Task 2.ipynb`: Jupyter notebook containing code for data preprocessing, model development, training, and evaluation
- `finallyTrained.h5`: Trained model weights saved in H5 format (Keras/TensorFlow)
- `classification_report (1).json`: Detailed classification metrics for model performance evaluation
- `Confusion Matrix.jpg`: Visualization of model predictions vs actual values
- `f1_scores.txt`: Summary of F1 performance metrics
- `Report.pdf`: Comprehensive project report including detailed analysis and findings
- `ONT_LAB Task 2.rar`: Archive containing project files

## Methodology
This project explored three different deep learning architectures, all enhanced with attention mechanisms:

1. **LSTM with attention**
2. **GRU with attention**
3. **Temporal Convolutional Neural Network (TCN) with attention**

Each model was trained and evaluated based on accuracy, precision, recall, F1-score, and training/evaluation time to determine the optimal architecture for the classification task.

## Model Performance
The final model demonstrates exceptional classification accuracy across all four classes:

### F1 Scores by Class:
- Class 0: 0.9974 (Support: 16,389 samples)
- Class 1: 0.9821 (Support: 4,048 samples)
- Class 2: 0.9808 (Support: 5,380 samples)
- Class 3: 0.9951 (Support: 15,100 samples)

### Overall Metrics:
- **Accuracy**: 0.9929
- **Macro F1 Score**: 0.9888
- **Micro F1 Score**: 0.9929
- **Weighted F1 Score**: 0.9929

These results significantly exceed the target accuracy of 95%, demonstrating the effectiveness of the chosen approach and attention mechanisms in improving model performance.

## Model Architecture
The final model was trained and saved as a Keras/TensorFlow model. It incorporates attention mechanisms to better capture temporal dependencies in the time series data. For specific architecture details, hyperparameters, and comparative analysis between LSTM, GRU, and TCN implementations, please refer to the Jupyter notebook and the detailed project report.

## Data Preprocessing Steps
1. Data inspection and cleaning (handling missing values, outliers)
2. Normalization/standardization of the time series
3. Feature engineering to extract useful patterns from the time series data
4. Data splitting into training, validation, and test sets

## Usage
To use the trained model:
1. Load the model using the Keras API:
```python
from tensorflow.keras.models import load_model
model = load_model('finallyTrained.h5')
```
2. Prepare your input data (ensure it's preprocessed the same way as during training)
3. Make predictions:
```python
predictions = model.predict(input_data)
```

## Requirements
- Python 3.x
- TensorFlow/Keras
- Numpy
- Matplotlib (for visualization)
- Scikit-learn (for metrics calculation)
- Pandas (for data manipulation)

## Key Findings
- The implementation of attention mechanisms significantly improved the performance of all three model architectures
- Comparative analysis showed differences in training time and performance metrics between LSTM, GRU, and TCN models
- The final model successfully achieved over 99% accuracy, exceeding the 95% target

For detailed comparisons between the different architectures and comprehensive evaluation results, please refer to the Report.pdf document.