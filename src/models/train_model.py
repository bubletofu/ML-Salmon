import os
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append('/Users/phuong/Desktop/hcmut/242/ML-Salmon/src')
from HMM import HMM  

class Train:
    
    def __init__(self, X_train_tfidf, y_train, model=None):
        self.X_train_tfidf = X_train_tfidf
        self.y_train = y_train
        self.model = model
        self.predictions = None

    def calculate_confusion_matrix(self):
        return confusion_matrix(self.y_train, self.predictions)
    
    def calculate_metrics(self):
        precision = precision_score(self.y_train, self.predictions)
        recall = recall_score(self.y_train, self.predictions)
        f1 = f1_score(self.y_train, self.predictions)
        return precision, recall, f1
    
    def generate_classification_report(self):
        return classification_report(self.y_train, self.predictions)
    
    def calculate_roc_auc_score(self):
        try:
            return roc_auc_score(self.y_train, self.predictions)
        except ValueError:  # Handle case when ROC AUC cannot be computed
            return None
        
    def train_svm_model(self, X_train_tfidf, y_train, param_grid=None):
        """
        Trains a Support Vector Machine (SVM) using the preprocessed data for sentiment analysis.
        
        Args:
            param_grid: Hyperparameter grid for grid search (default=None)
        
        Returns:
            A trained SVM model
        """
        # Default hyperparameters if none are passed
        if param_grid is None:
            param_grid = {'C': [0.1, 1.0, 10.0]}

        svm_model = LinearSVC(random_state=42, max_iter=1000)
        
        grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_tfidf, y_train)
        
        self.model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")
        
        self.predictions = self.model.predict(X_train_tfidf)
        
        self.accuracy = np.sum(self.predictions == y_train) / len(y_train)
        
        print(f"Model trained with accuracy: {self.accuracy}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_train, self.predictions)}")
        print(f"Classification Report:\n{classification_report(y_train, self.predictions)}")
        
        self.roc_auc_score = self.calculate_roc_auc_score()
        print(f"ROC AUC Score: {self.roc_auc_score}")
        

        
        return self.model

    def train_hidden_markov_model(self, X_train_tfidf, y_train, n_components=2, random_state=42):
        """
        Trains a Hidden Markov Model (HMM) using the preprocessed data for sentiment analysis.
        
        Args:
            n_components: Number of hidden states in the HMM (default=3)
            random_state: Random seed for reproducibility
        
        Returns:
            A trained HMM instance
        """
        # Initialize custom HMM model
        hmm_model = HMM(n_components=n_components, n_features=X_train_tfidf.shape[1], random_state=random_state)
        
        # Convert sparse matrix to dense array (necessary for HMM)
        X_train_dense = X_train_tfidf.toarray()  # Convert sparse matrix to dense numpy array
        
        # Train the model
        hmm_model.fit(X_train_dense, max_iter=100)
        
        self.model = hmm_model
        
        self.predictions = self.model.predict(X_train_dense)
        
        # Compute the accuracy and evaluation metrics
        self.accuracy = np.sum(self.predictions == y_train) / len(y_train)
        
        # Check the confusion matrix to understand model performance
        print(confusion_matrix(y_train, self.predictions))
        
        # Calculate precision, recall, and F1 score
        self.precision = precision_score(y_train, self.predictions, zero_division=1)
        self.recall = recall_score(y_train, self.predictions)
        self.f1_score = f1_score(y_train, self.predictions)
        
        self.classification_report = classification_report(y_train, self.predictions)
        
        self.roc_auc_score = self.calculate_roc_auc_score()
        
        print(f"Model trained with accuracy: {self.accuracy}")
        print(f"Classification Report:\n{self.classification_report}")
        return self.model
    
    def save_model(self, model_path):
        directory = os.path.dirname(model_path)
        if not os.path.exists(directory):
            os.makedirs(directory)  

        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        return model_path