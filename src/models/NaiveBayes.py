import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report, roc_auc_score

class BayesianClassifier:
    def __init__(self):
        self.model = GaussianNB()

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
        train_data_dir = r"/content/ML-Salmon/data/aclImdb/train"
        data = BayesianClassifier.load_data(train_data_dir)

        # Tiền xử lý dữ liệu
        X = data['review']
        y = data['label']
        X = X.apply(len).values.reshape(-1, 1)

        bc = BayesianClassifier()
        bc.train(X, y)
        print("Evaluation Metrics:", bc.evaluate())

        model_save_path = r"/content/ML-Salmon/models_trained/trained_bayesian.pkl"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        bc.save_model(model_save_path)
        print(f"Model saved to {model_save_path}")

# Gọi main() khi chạy file
if __name__ == '__main__':
    BayesianClassifier.main()
