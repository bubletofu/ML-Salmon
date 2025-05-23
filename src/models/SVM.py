import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualization.visualize import generate_visualizations
from train_model import Train

# Define base data path
BASE_DATA_PATH = "data/aclImdb"

def load_data(data_path):
    """
    Load text data from positive and negative subdirectories.
    Args:
        data_path: Path to the data directory (e.g., 'data/aclImdb/train')
    Returns:
        texts: List of text samples
        labels: List of corresponding labels (1 for positive, 0 for negative)
    """
    texts, labels = [], []
    label_map = {"pos": 1, "neg": 0}  # Positive: 1, Negative: 0
    for subdir in ["pos", "neg"]:
        subdir_path = os.path.join(data_path, subdir)
        if not os.path.exists(subdir_path):
            raise FileNotFoundError(f"Directory not found: {subdir_path}")
        for file_name in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    if content:
                        texts.append(content)
                        labels.append(label_map[subdir])
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    if not texts:
        raise ValueError("No valid files found in the provided directory")
    return texts, labels

# Load training and test data
try:
    train_texts, train_labels = load_data(os.path.join(BASE_DATA_PATH, "train"))
    test_texts, test_labels = load_data(os.path.join(BASE_DATA_PATH, "test"))
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Split training data into train and validation sets
X_train_texts, X_val_texts, y_train, y_val = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)

# Vectorize the text data using TfidfVectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=15000,
    ngram_range=(1, 2),  # Include unigrams and bigrams
    lowercase=True
)
X_train_tfidf = vectorizer.fit_transform(X_train_texts)
X_val_tfidf = vectorizer.transform(X_val_texts)
X_test_tfidf = vectorizer.transform(test_texts)

# Initialize and train the model using the Train class
trainer = Train(X_train_tfidf, y_train)
param_grid = {'C': [0.1, 1.0, 10.0]}  # Define hyperparameter grid
svm_model = trainer.train_svm_model(X_train_tfidf, y_train, param_grid)

# Save the trained model and vectorizer
os.makedirs("models_trained", exist_ok=True)
trainer.save_model("models_trained/trained_svm_model.pkl")
joblib.dump(vectorizer, "models_trained/svm_vectorizer.pkl")
print("Vectorizer saved to models_trained/svm_vectorizer.pkl")

# Evaluate on validation set
val_predictions = svm_model.predict(X_val_tfidf)
print("Validation Performance:")
print(classification_report(y_val, val_predictions))
print("Confusion Matrix (Validation):")
print(confusion_matrix(y_val, val_predictions))
if len(np.unique(y_val)) == 2:
    val_scores = svm_model.decision_function(X_val_tfidf)
    print(f"ROC-AUC Score (Validation): {roc_auc_score(y_val, val_scores):.4f}")

# Evaluate on test set
test_predictions = svm_model.predict(X_test_tfidf)
print("\nTest Performance:")
print(classification_report(test_labels, test_predictions))
print("Confusion Matrix (Test):")
print(confusion_matrix(test_labels, test_predictions))
if len(np.unique(test_labels)) == 2:
    test_scores = svm_model.decision_function(X_test_tfidf)
    print(f"ROC-AUC Score (Test): {roc_auc_score(test_labels, test_scores):.4f}")

# Predicting function
def predict_sentiment(texts, model_path="models_trained/trained_svm_model.pkl", 
                     vectorizer_path="models_trained/svm_vectorizer.pkl"):
    """
    Predict sentiment for a list of texts.
    Args:
        texts: List of text strings to classify
        model_path: Path to the saved SVM model
        vectorizer_path: Path to the saved vectorizer
    Returns:
        List of predicted sentiments (Positive/Negative)
    """
    try:
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
    except FileNotFoundError as e:
        print(f"Error loading model or vectorizer: {e}")
        return None
    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    label_map = {1: "Positive", 0: "Negative"}
    return [label_map[pred] for pred in predictions]

svm_model = joblib.load("models_trained/trained_svm_model.pkl")
vectorizer = joblib.load("models_trained/svm_vectorizer.pkl")
generate_visualizations(svm_model, X_test_tfidf, test_labels, output_dir="models_trained/plots")