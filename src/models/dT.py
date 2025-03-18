import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report, roc_auc_score

class DecisionTree:
    def __init__(self, max_depth=None, random_state=None):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.X_test = X_test
        self.y_test = y_test

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self):
        y_pred = self.predict(self.X_test)
        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, average='weighted'),
            "recall": recall_score(self.y_test, y_pred, average='weighted'),
            "f1_score": f1_score(self.y_test, y_pred, average='weighted'),
            "classification_report": classification_report(self.y_test, y_pred),
            # Use probabilities for the positive class (class 1) for roc_auc_score
            "roc_auc": roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        }
        return metrics

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        self.model = joblib.load(filepath)

    @staticmethod
    def load_data(data_dir):
        data = []
        labels = []
        for label, folder in enumerate(['neg', 'pos']):
            folder_path = os.path.join(data_dir, folder)
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data.append(file.read())
                    labels.append(label)
        return pd.DataFrame({'review': data, 'label': labels})

    @staticmethod
    def main():
        # Load data from aclImdb/train
        train_data_dir = r"C:\Users\ASUS\Documents\MachineLearning\ML-Salmon\ML-Salmon\data\aclImdb\train"
        data = DecisionTree.load_data(train_data_dir)

        # Preprocess data
        X = data['review']
        y = data['label']

        # Example: Convert text data to simple numerical features (e.g., length of reviews)
        X = X.apply(len).values.reshape(-1, 1)

        # Train and evaluate the model
        dt = DecisionTree(max_depth=3)
        dt.train(X, y)
        print("Evaluation Metrics:", dt.evaluate())

        # Save the trained model
        model_save_path = r"C:\Users\ASUS\Documents\MachineLearning\ML-Salmon\ML-Salmon\models_trained\trained_decision_tree.pkl"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure the directory exists
        dt.save_model(model_save_path)
        print(f"Model saved to {model_save_path}")

# Call main() outside the class definition
if __name__ == '__main__':
    DecisionTree.main()