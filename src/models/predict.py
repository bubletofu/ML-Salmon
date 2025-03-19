import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Load the trained model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "..", "models_trained", "trained_decision_tree.pkl")
print(f"Resolved model path: {model_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the file exists.")

model = joblib.load(model_path)

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

# Preprocess the test data (same as training: convert to length of reviews)
X_test_processed = np.array([len(review) for review in X_test]).reshape(-1, 1)

# Ensure labels are encoded (though already 0 and 1, this ensures consistency)
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

# Make predictions
y_pred = model.predict(X_test_processed)

# Evaluate the model with a classification report
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred, zero_division=0))

# Optional: Print additional metrics for detailed analysis
accuracy = (y_pred == y_test_encoded).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")