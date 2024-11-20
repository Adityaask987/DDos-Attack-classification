# DDoS Attack Classification Using Machine Learning

This project aims to classify DDoS attacks using machine learning techniques. The dataset contains network traffic data, which is analyzed and modeled to distinguish between normal traffic and various types of DDoS attacks.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Usage](#usage)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ddos-attack-classification.git
   cd ddos-attack-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have TensorFlow installed. For GPU support:
   ```bash
   pip install tensorflow-gpu
   ```

---

## Dataset

The dataset contains network traffic data with features such as:
- **Source IP**
- **Destination IP**
- **Packet Size**
- **Flags**

### Labels:
- **Normal traffic**: Labeled as `0`
- **DDoS attack traffic**: Labeled as `1`

The dataset is preprocessed, including feature scaling and encoding, before training.

---

## Approach

### Models Used:
1. **Logistic Regression**:
   - A baseline model to classify DDoS attacks.
   - Evaluated with metrics like accuracy, precision, recall, and F1-score.

2. **Neural Network**:
   - A deep learning model built with TensorFlow/Keras.
   - Architecture:
     - Input Layer: Corresponding to the number of features in the dataset.
     - Hidden Layers: Two layers with `ReLU` activation.
     - Output Layer: Single neuron with `sigmoid` activation for binary classification.
   - Model compiled using:
     - Loss: Binary Cross-Entropy
     - Optimizer: Adam
     - Metrics: Accuracy

---

## Results

1. **Logistic Regression**:
   - Accuracy: ~XX% (Replace with actual results)
   - Precision: ~XX%
   - Recall: ~XX%
   - F1-Score: ~XX%

2. **Neural Network**:
   - Training Accuracy: ~XX%
   - Test Accuracy: ~XX%

3. **Confusion Matrix**:
   ```
   [[True Negative, False Positive],
    [False Negative, True Positive]]
   ```

---

## Usage

1. Train the models:
   - Logistic Regression:
     ```python
     logModel.fit(X_train, y_train)
     ```
   - Neural Network:
     ```python
     model.fit(X_train, y_train, epochs=5, batch_size=32)
     ```

2. Evaluate the models:
   - Accuracy on test set:
     ```python
     accuracy_score(y_test, y_pred)
     ```
   - Confusion Matrix:
     ```python
     from sklearn.metrics import confusion_matrix
     confusion_matrix(y_test, y_pred)
     ```

3. Save and load models for future use:
   ```python
   model.save('ddos_model.h5')
   loaded_model = tf.keras.models.load_model('ddos_model.h5')
   ```

---

## Contributing

Contributions are welcome! Fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
