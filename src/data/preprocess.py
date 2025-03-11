import os
import urllib.request
import tarfile
import pandas as pd
import string
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

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

class DataPreprocessing:
    def __init__(self, data_dir="data/", dataset_url="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", dataset_filename="aclImdb_v1.tar.gz"):
        self.data_dir = data_dir
        self.dataset_url = dataset_url
        self.dataset_filename = dataset_filename
        self.dataset_path = os.path.join(data_dir, dataset_filename)
        self.extracted_path = os.path.join(data_dir, "aclImdb")
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    # Step 1: Download Dataset
    def download_dataset(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if not os.path.exists(self.dataset_path):
            print(f"Downloading dataset from {self.dataset_url}...")
            urllib.request.urlretrieve(self.dataset_url, self.dataset_path)
            print("Download complete!")
        else:
            print("Dataset already exists, skipping download.")

    # Step 2: Unzip the Dataset
    def unzip_dataset(self):
        if not os.path.exists(self.extracted_path):
            print(f"Extracting {self.dataset_filename}...")
            with tarfile.open(self.dataset_path, "r:gz") as tar:
                tar.extractall(path=self.data_dir)
            print("Extraction complete!")
        else:
            print("Dataset already extracted, skipping extraction.")

    # Step 3: Load and Split Sentiment Data
    def load_sentiment_data(self, directory, sentiment):
        data = []
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
                    data.append([text, sentiment])
        return pd.DataFrame(data, columns=['text', 'sentiment'])

    def split_data(self):
        data_dir = os.path.abspath(self.extracted_path)
        train_pos_dir = os.path.join(data_dir, 'train', 'pos')
        train_neg_dir = os.path.join(data_dir, 'train', 'neg')
        train_pos_df = self.load_sentiment_data(train_pos_dir, 1)
        train_neg_df = self.load_sentiment_data(train_neg_dir, 0)

        self.train_df = pd.concat([train_pos_df, train_neg_df], ignore_index=True)

        test_pos_dir = os.path.join(data_dir, 'test', 'pos')
        test_neg_dir = os.path.join(data_dir, 'test', 'neg')
        test_pos_df = self.load_sentiment_data(test_pos_dir, 1)
        test_neg_df = self.load_sentiment_data(test_neg_dir, 0)

        self.test_df = pd.concat([test_pos_df, test_neg_df], ignore_index=True)

        self.train_df = self.train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.test_df = self.test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Step 4: Preprocess Text Data
    def text_lowercase(self, text):
        return text.lower()

    def remove_html_tags(self, text):
        return re.sub(r'<.*?>', '', text)

    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_stopwords(self, words):
        return [word for word in words if word.lower() not in self.stop_words]

    def lemmatize(self, words):
        return [self.lemmatizer.lemmatize(word) for word in words]

    def clean_word(self, words):
        return [word for word in words if re.match(r'^[a-zA-Z\-]+$', word)]

    def decontracted(self, phrase):
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can't", "can not", phrase)
        phrase = re.sub(r"n't", " not", phrase)
        phrase = re.sub(r"'re", " are", phrase)
        phrase = re.sub(r"'s", " is", phrase)
        phrase = re.sub(r"'d", " would", phrase)
        phrase = re.sub(r"'ll", " will", phrase)
        phrase = re.sub(r"'t", " not", phrase)
        phrase = re.sub(r"'ve", " have", phrase)
        phrase = re.sub(r"'m", " am", phrase)
        return phrase

    def clean_text(self, text):
        text = self.remove_html_tags(text)
        text = self.decontracted(text)
        text = self.text_lowercase(text)
        text = self.remove_punctuation(text)
        words = self.tokenize(text)
        words = self.clean_word(words)
        words = self.remove_stopwords(words)
        words = self.lemmatize(words)
        return ' '.join(words)

    def clean_data(self):
        self.split_data()
        self.train_df['text'] = self.train_df['text'].apply(self.clean_text)
        self.test_df['text'] = self.test_df['text'].apply(self.clean_text)

    # Step 5: Vectorize the Data
    def vectorization(self):
        self.clean_data()
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(self.train_df['text'])
        X_test_tfidf = vectorizer.transform(self.test_df['text'])
        y_train = self.train_df['sentiment']
        y_test = self.test_df['sentiment']
        return X_train_tfidf, X_test_tfidf, y_train, y_test



    # Step 7: Save the trained model
    def save_model(self, model_path):
        directory = os.path.dirname(model_path)
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist

        # Save the model to the specified path
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        return model_path

    # Step 8: Execute the complete pipeline
    def execute_pipeline(self):
        """
        Executes the full pipeline: downloading, preprocessing, training, and saving the model.
        """
        #Download and unzip the dataset if necessary
        self.download_dataset() 
        self.unzip_dataset()

        # Vectorize the data and split into training and testing datasets
        X_train_tfidf, X_test_tfidf, y_train, y_test = self.vectorization()

        # Initialize and train the model
        train_model = Train(X_train_tfidf, y_train)  # Assuming Train class is defined
        trained_model = train_model.train_hidden_markov_model(X_train_tfidf, y_train, n_components=3, random_state=42)

        # Save the trained model
        model_path = train_model.save_model("models_trained/trained_hmm_model.pkl")

        print("Training complete. Model saved at:", model_path)
        return trained_model


data_preprocessor = DataPreprocessing()
trained_model = data_preprocessor.execute_pipeline()