# ONT_LAB Task 2: Multi-Class Classification Project

## Project Overview
This project implements a multi-class classification model capable of accurately categorizing inputs into four distinct classes. The model achieves excellent performance with high precision, recall, and F1 scores across all classes.

## Project Structure
- `Task 2.ipynb`: Jupyter notebook containing the project code, data preprocessing, model training, and evaluation
- `finallyTrained.h5`: Trained model weights saved in H5 format (Keras/TensorFlow)
- `classification_report (1).json`: Detailed classification metrics for model performance evaluation
- `Confusion Matrix.jpg`: Visualization of model predictions vs actual values
- `f1_scores.txt`: Summary of F1 performance metrics
- `Report.pdf`: Comprehensive project report

## Model Performance
The model demonstrates excellent classification accuracy across all four classes:

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

## Model Architecture
The model was trained and saved as a Keras/TensorFlow model. It was designed to handle a multi-class classification problem with 4 output classes. For specific architecture details and hyperparameters, please refer to the Jupyter notebook and project report.

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

## License
[Include appropriate license information here]

## Author
[Include author/team information here]