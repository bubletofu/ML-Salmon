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