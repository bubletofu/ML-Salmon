import joblib
import numpy as np
import os
import logging
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from hmmlearn.hmm import MultinomialHMM
import sys

sys.path.append('/Users/phuong/Desktop/hcmut/242/ML-Salmon/src')
from models.HMM import HMM

class Train:
    
    def __init__(self, X_train_tfidf, y_train, model=None):
        self.X_train_tfidf = X_train_tfidf
        self.y_train = y_train
        self.model = model
        logging.basicConfig(level=logging.INFO)

    def calculate_confusion_matrix(self):
        if not hasattr(self, 'predictions') or self.predictions is None:
            logging.error("Predictions not available. Please ensure the model is trained.")
            return None
        return confusion_matrix(self.y_train, self.predictions)
    
    def calculate_metrics(self):
        precision = precision_score(self.y_train, self.predictions, zero_division=1)
        recall = recall_score(self.y_train, self.predictions)
        f1 = f1_score(self.y_train, self.predictions)
        return precision, recall, f1
    
    def generate_classification_report(self):
        return classification_report(self.y_train, self.predictions)
    
    def calculate_roc_auc_score(self):
        try:
            return roc_auc_score(self.y_train, self.predictions)
        except ValueError:
            logging.warning("ROC AUC cannot be computed, possibly due to class imbalance.")
            return None
        
    def train_hidden_markov_model(self, X_train_tfidf, y_train, n_components=2, random_state=42):
        """
        Trains a Hidden Markov Model on the provided training data.
        
        Args:
            X_train_tfidf (array-like): The training data in TF-IDF format.
            y_train (array-like): The sentiment labels for the training data.
            n_components (int): The number of hidden states in the HMM.
            random_state (int): The random seed for reproducibility.
            
        Returns:
            trained_model (HMM), predictions (array): The trained Hidden Markov Model and its predictions.
        """
        self.model = MultinomialHMM(n_components=n_components, random_state=random_state)
        
        # Convert TF-IDF values to integer counts by rounding
        X_train_counts = X_train_tfidf.toarray().round().astype(int)  # Convert to integers
        
        # Fit the model using the transformed training data
        self.model.fit(X_train_counts)
        
        # Make predictions on the training set
        self.predictions = self.model.predict(X_train_counts)
        
        logging.info("Hidden Markov Model trained successfully.")
        
        return self.model, self.predictions


    def save_model(self, model_path, compress_level=3):
        try:
            directory = os.path.dirname(model_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # Save the model with compression level (default is 3)
            joblib.dump(self.model, model_path, compress=compress_level)  # Use compression level
            
            logging.info(f"Model saved to {model_path}")
        except IOError as e:
            logging.error(f"Error saving the model: {e}")
            raise
        return model_path
