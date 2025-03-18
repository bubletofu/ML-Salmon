import os
import urllib.request
import tarfile
from numpy import vectorize
import pandas as pd
import logging
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from hmmlearn.hmm import MultinomialHMM
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, roc_auc_score
import joblib
import sys

sys.path.append('/Users/phuong/Desktop/hcmut/242/ML-Salmon/src')
from models.train_model import Train
from visualization.visualize import plot_sentiment_distribution, plot_top_tfidf_features, plot_confusion_matrix, plot_roc_curve, plot_learning_curves


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

class DataPreprocessing:
    def __init__(self, data_dir="data/", dataset_url="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", dataset_filename="aclImdb_v1.tar.gz"):
        """
        Initializes DataPreprocessing with given parameters.

        Args:
            data_dir (str): Directory where the dataset will be saved.
            dataset_url (str): URL to download the dataset from.
            dataset_filename (str): Name of the dataset file.
        """
        self.data_dir = data_dir
        self.dataset_url = dataset_url
        self.dataset_filename = dataset_filename
        self.dataset_path = os.path.join(data_dir, dataset_filename)
        self.extracted_path = os.path.join(data_dir, "aclImdb")
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def download_dataset(self):
        """Downloads the dataset if it does not exist locally."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if not os.path.exists(self.dataset_path):
            self.logger.info(f"Downloading dataset from {self.dataset_url}...")
            urllib.request.urlretrieve(self.dataset_url, self.dataset_path)
            self.logger.info("Download complete!")
        else:
            self.logger.info("Dataset already exists, skipping download.")

    def unzip_dataset(self):
        """Extracts the dataset if it has not been extracted yet."""
        if not os.path.exists(self.extracted_path):
            self.logger.info(f"Extracting {self.dataset_filename}...")
            with tarfile.open(self.dataset_path, "r:gz") as tar:
                tar.extractall(path=self.data_dir)
            self.logger.info("Extraction complete!")
        else:
            self.logger.info("Dataset already extracted, skipping extraction.")

    def load_sentiment_data(self, directory, sentiment):
        """Loads sentiment data from the given directory."""
        data = []
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
                    data.append([text, sentiment])
        return pd.DataFrame(data, columns=['text', 'sentiment'])

    def split_data(self):
        """Splits the dataset into training and testing data."""
        data_dir = os.path.abspath(self.extracted_path)
        
        # Load training data
        train_pos_df = self.load_sentiment_data(os.path.join(data_dir, 'train', 'pos'), 1)
        train_neg_df = self.load_sentiment_data(os.path.join(data_dir, 'train', 'neg'), 0)
        self.train_df = pd.concat([train_pos_df, train_neg_df], ignore_index=True)

        # Load test data
        test_pos_df = self.load_sentiment_data(os.path.join(data_dir, 'test', 'pos'), 1)
        test_neg_df = self.load_sentiment_data(os.path.join(data_dir, 'test', 'neg'), 0)
        self.test_df = pd.concat([test_pos_df, test_neg_df], ignore_index=True)

        # Shuffle datasets
        self.train_df = self.train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.test_df = self.test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    def clean_text(self, text):
        text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Only keep alphabets and spaces
        text = text.lower()
        words = word_tokenize(text)
        words = [word for word in words if word not in self.stop_words]  # Remove stopwords
        words = [self.lemmatizer.lemmatize(word) for word in words]  # Lemmatization
        return " ".join(words)
    
    def clean_data(self):
        """Cleans the text data for both training and testing sets."""
        self.split_data()
        self.train_df['text'] = self.train_df['text'].apply(self.clean_text)
        self.test_df['text'] = self.test_df['text'].apply(self.clean_text)

    def vectorization(self):
        self.clean_data()  # Ensure the data is cleaned before vectorization
        # vectorizer = TfidfVectorizer(max_features=10000) 
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)  # Bigrams

        X_train_tfidf = vectorizer.fit_transform(self.train_df['text'])
        X_test_tfidf = vectorizer.transform(self.test_df['text'])
        y_train = self.train_df['sentiment']
        y_test = self.test_df['sentiment']
        return X_train_tfidf, X_test_tfidf, y_train, y_test
    




    def execute_pipeline(self):
        """Executes the full pipeline: downloading, preprocessing, training, and visualizing."""
        # Download and unzip the dataset
        self.download_dataset() 
        self.unzip_dataset()

        # Vectorize the data and split into training and testing datasets
        X_train_tfidf, X_test_tfidf, y_train, y_test = self.vectorization()

        # Initialize and train the model
        train_model = Train(X_train_tfidf, y_train)  # Create an instance of the Train class
        trained_model = train_model.train_hidden_markov_model(X_train_tfidf, y_train, n_components=2, random_state=42)  # Call the method on the instance

        # Visualize sentiment distribution (before training)
        plot_sentiment_distribution(self.train_df)

        # Visualize top TF-IDF features
        plot_top_tfidf_features(X_train_tfidf, vectorize)

        # Visualize confusion matrix and ROC curve after predictions
        plot_confusion_matrix(y_train, trained_model.predictions)
        plot_roc_curve(y_train, trained_model.predictions)

        # Save the trained model
        model_path = train_model.save_model("models_trained/trained_hmm_model.pkl")

        print("Training complete. Model saved at:", model_path)
        return trained_model

# Running the Data Preprocessing and Model Training Pipeline
data_preprocessor = DataPreprocessing()
trained_model = data_preprocessor.execute_pipeline()

