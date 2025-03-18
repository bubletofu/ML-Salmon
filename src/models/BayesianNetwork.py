import os
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report, roc_auc_score

class BayesianGraph:
    def __init__(self, k_features=80):
        self.k_features = k_features
        self.cv = None
        self.selector = None
        self.selected_features = None
        self.trained_model = None
        self.infer = None
        self.X_test = None
        self.y_test = None

    def train(self, X, y):
        # 1. Trích xuất n-gram từ text sử dụng CountVectorizer
        self.cv = CountVectorizer(ngram_range=(1, 2), stop_words='english')
        X_counts = self.cv.fit_transform(X)

        # 2. Lựa chọn k đặc trưng hàng đầu theo chi-square test
        self.selector = SelectKBest(chi2, k=self.k_features)
        X_selected = self.selector.fit_transform(X_counts, y)
        selected_indices = self.selector.get_support(indices=True)
        self.selected_features = [self.cv.get_feature_names_out()[i] for i in selected_indices]

        X_features = (X_selected > 0).astype(int)
        df_features = pd.DataFrame(X_features.toarray(), columns=self.selected_features)
        df_features['label'] = y.values

        train_df, test_df = train_test_split(df_features, test_size=0.2, random_state=42, stratify=df_features['label'])

        hc = HillClimbSearch(train_df)
        best_structure = hc.estimate(scoring_method=BicScore(train_df))
        self.trained_model = BayesianNetwork(best_structure.edges())
        self.trained_model.fit(train_df, estimator=BayesianEstimator, prior_type='BDeu')
        
        self.infer = VariableElimination(self.trained_model)
        

        self.X_test = test_df.drop(columns=['label'])
        self.y_test = test_df['label']

    def predict(self, X_features):
        """
        Dự đoán nhãn cho tập dữ liệu đầu vào X_features (DataFrame với các cột là selected_features).
        Với mỗi mẫu, thực hiện suy diễn xác suất của biến 'label' và chọn nhãn có xác suất cao nhất.
        """
        predictions = []
        for _, row in X_features.iterrows():
            evidence = row.to_dict()
            query_result = self.infer.query(variables=['label'], evidence=evidence)
            pred_label = query_result.values.argmax()
            predictions.append(pred_label)
        return np.array(predictions)

    def evaluate(self):
        y_pred = self.predict(self.X_test)
        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, average='weighted'),
            "recall": recall_score(self.y_test, y_pred, average='weighted'),
            "f1_score": f1_score(self.y_test, y_pred, average='weighted'),
            "classification_report": classification_report(self.y_test, y_pred)
        }
        if len(np.unique(self.y_test)) == 2:
            y_prob = []
            for _, row in self.X_test.iterrows():
                evidence = row.to_dict()
                query_result = self.infer.query(variables=['label'], evidence=evidence)
                # Giả sử lớp 1 là positive
                prob1 = query_result.values[1]
                y_prob.append(prob1)
            metrics["roc_auc"] = roc_auc_score(self.y_test, y_prob)
        return metrics

    def save_model(self, filepath):
        artifacts = {
            'trained_model': self.trained_model,
            'count_vectorizer': self.cv,
            'selector': self.selector,
            'selected_features': self.selected_features
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(artifacts, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            artifacts = pickle.load(f)
        self.trained_model = artifacts['trained_model']
        self.cv = artifacts['count_vectorizer']
        self.selector = artifacts['selector']
        self.selected_features = artifacts['selected_features']
        self.infer = VariableElimination(self.trained_model)

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
        data = BayesianGraph.load_data(train_data_dir)

        X = data['review']
        y = data['label']

        bg = BayesianGraph(k_features=80)
        bg.train(X, y)
        print("Evaluation Metrics:", bg.evaluate())

        model_save_path = r"/content/ML-Salmon/models_trained/trained_bayesian_network.pkl"
        bg.save_model(model_save_path)
        print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    BayesianGraph.main()
