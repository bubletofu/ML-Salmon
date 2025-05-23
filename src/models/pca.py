import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualization.visualize import generate_visualizations

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
    dir = r"/Users/thuckhue/Desktop/HCMUT/Machine_Learning/ML-Salmon/data/aclImdb"
    try:
        train_texts, train_labels = load_data(os.path.join(dir, "train"))
        test_texts, test_labels = load_data(os.path.join(dir, "test"))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    # Split training data
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
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
    X_val = vectorizer.transform(X_test_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    # Standardize the data
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    # X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_tfidf)

    # Apply PCA
    pca_model = PCA(n_components=2)
    X_train_pca = pca_model.fit_transform(X_train_scaled.toarray())  

    # Save models
    os.makedirs("models_trained", exist_ok=True)
    joblib.dump(pca_model, "models_trained/pca_model.pkl")
    joblib.dump(vectorizer, "models_trained/pca_vectorizer.pkl")
    joblib.dump(scaler, "models_trained/pca_scaler.pkl")

    # Using Logistic Regression for classification
    # Split the PCA-transformed data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_pca, y_train, test_size=0.2, random_state=42
    )
    clf = LogisticRegression()
    clf.fit(X_train_split, y_train_split)

    # Transform test data using PCA
    X_test_pca = pca_model.transform(X_test_scaled.toarray())
    y_pred = clf.predict(X_test_pca)
    accuracy = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred, average='weighted')

    print("Accuracy:", round(accuracy, 4))
    print("F1 Score:", round(f1, 4))

    cm = confusion_matrix(test_labels, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()
    
if __name__ == "__main__":
    pca()