""" # test NeuralNetwork
import pickle
import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "..", "models_trained", "trained_neural_network.pkl")
print(f"Resolved model path: {model_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the file exists.")

# Load the model and tokenizer from the .pkl file
with open(model_path, "rb") as file:
    model_data = pickle.load(file)

# Extract the model and tokenizer
model = model_data['model']  # Trích xuất mô hình Keras
tokenizer = model_data['tokenizer']  # Trích xuất tokenizer

# Load and preprocess the test dataset
test_dir = os.path.join(script_dir, "..", "..", "data", "aclImdb", "test")
print(f"Resolved test directory: {test_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory not found at {test_dir}")

def load_reviews_and_labels(directory):
    reviews = []
    labels = []
    for label in ["pos", "neg"]:
        label_dir = os.path.join(directory, label)
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Label directory not found: {label_dir}")
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                reviews.append(file.read())
            labels.append(1 if label == "pos" else 0)
    return reviews, labels

X_test, y_test = load_reviews_and_labels(test_dir)

# Preprocess the test data using the same tokenizer and padding as during training
max_length = 100  # Phải khớp với giá trị trong train_model.py
padding_type = 'post'  # Phải khớp với giá trị trong train_model.py
trunc_type = 'post'  # Phải khớp với giá trị trong train_model.py

# Tokenize and pad the test reviews
sequences = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Ensure labels are encoded
label_encoder = LabelEncoder()
y_test = label_encoder.fit_transform(y_test)

# Make predictions
y_pred = model.predict(X_test_padded)  # Dự đoán xác suất
y_pred_classes = (y_pred > 0.5).astype(int).flatten()  # Chuyển xác suất thành nhãn lớp (0 hoặc 1)

# Evaluate the model with a classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes, zero_division=0))
"""
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from pgmpy.inference import VariableElimination

# Load the trained model and related artifacts
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "..", "models_trained", "trained_bayesian_network.pkl")
print(f"Resolved model path: {model_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the file exists.")

# Load the model and artifacts from the .pkl file
with open(model_path, "rb") as file:
    artifacts = pickle.load(file)

# Extract components
trained_model = artifacts['trained_model']  # BayesianNetwork
count_vectorizer = artifacts['count_vectorizer']  # CountVectorizer
selector = artifacts['selector']  # SelectKBest
selected_features = artifacts['selected_features']  # List of selected feature names

# Initialize inference engine
infer = VariableElimination(trained_model)

# Load and preprocess the test dataset
test_dir = os.path.join(script_dir, "..", "..", "data", "aclImdb", "test")
print(f"Resolved test directory: {test_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory not found at {test_dir}")

def load_reviews_and_labels(directory):
    reviews = []
    labels = []
    for label in ["pos", "neg"]:
        label_dir = os.path.join(directory, label)
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Label directory not found: {label_dir}")
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                reviews.append(file.read())
            labels.append(1 if label == "pos" else 0)
    return reviews, labels

X_test, y_test = load_reviews_and_labels(test_dir)

# Preprocess the test data
X_counts = count_vectorizer.transform(X_test)  # Vectorize text with the same CountVectorizer
print(f"Shape of X_counts: {X_counts.shape}")  # Debugging: Check the shape of the vectorized data

# Ensure the number of features matches the training data
X_selected = selector.transform(X_counts)
print(f"Shape of X_selected: {X_selected.shape}")  # Debugging: Check the shape after feature selection

# Convert to binary presence (0 or 1)
X_features = (X_selected > 0).astype(int)

# Validate the shape of X_features against selected_features
if X_features.shape[1] != len(selected_features):
    raise ValueError(
        f"Shape mismatch: X_features has {X_features.shape[1]} features, "
        f"but selected_features has {len(selected_features)} features."
    )

# Create DataFrame with selected features
df_features = pd.DataFrame(X_features, columns=selected_features)

# Add temporary 'label' column for inference (will be ignored during prediction)
df_features['label'] = 0  # Placeholder, not used in prediction

# Make predictions
predictions = []
for _, row in df_features.iterrows():
    evidence = row.drop('label').to_dict()  # Remove temporary 'label' for inference
    query_result = infer.query(variables=['label'], evidence=evidence)
    pred_label = query_result.values.argmax()  # Take the class with highest probability
    predictions.append(pred_label)
y_pred = np.array(predictions)

# Evaluate the model with a classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))