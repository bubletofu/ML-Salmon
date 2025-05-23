import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from visualization.visualize import generate_visualizations
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define base data path
BASE_DATA_PATH = "data/aclImdb"

def load_data(data_path):
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

def pca():
    # Load training and test data
    try:
        train_texts, train_labels = load_data(os.path.join(BASE_DATA_PATH, "train"))
        test_texts, test_labels = load_data(os.path.join(BASE_DATA_PATH, "test"))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    # Split training data
    X_train_texts, X_val_texts, y_train, y_val = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42
    )

    # Vectorize the text data
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_texts)
    X_val = vectorizer.transform(X_val_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    # Standardize the data
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_tfidf)

    # Apply PCA
    pca_model = PCA(n_components=2)
    X_train_pca = pca_model.fit_transform(X_train_scaled.toarray())
    X_val_pca = pca_model.transform(X_val_scaled.toarray())
    X_test_pca = pca_model.transform(X_test_scaled.toarray())

    # Save models
    os.makedirs("models_trained", exist_ok=True)
    joblib.dump(pca_model, "models_trained/pca_model.pkl")
    joblib.dump(vectorizer, "models_trained/pca_vectorizer.pkl")
    joblib.dump(scaler, "models_trained/pca_scaler.pkl")

    # Visualization
    svm_model = joblib.load("models_trained/trained_svm_model.pkl")
    generate_visualizations(svm_model, X_test_pca, test_labels, output_dir="models_trained/plots")

def predict(texts, model_path='models_trained/pca_model.pkl', vectorizer_path='models_trained/pca_vectorizer.pkl', scaler_path='models_trained/pca_scaler.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model, vectorizer, or scaler file not found.")

    # Load vectorizer, scaler and PCA model
    vectorizer = joblib.load(vectorizer_path)
    scaler = joblib.load(scaler_path)
    pca_model = joblib.load(model_path)

    # Vectorize input texts
    X_tfidf = vectorizer.transform(texts)

    # Standardize
    X_scaled = scaler.transform(X_tfidf)

    # Apply PCA
    X_pca = pca_model.transform(X_scaled.toarray())

    return X_pca

if __name__ == "__main__":
    pca()