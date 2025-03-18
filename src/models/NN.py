import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, vocab_size=10000, embedding_dim=16, max_length=100, trunc_type='post', padding_type='post'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.trunc_type = trunc_type
        self.padding_type = padding_type
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.model = None

    def load_data(self, data_dir):
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

    def preprocess_data(self, data):
        reviews = data['review']
        labels = data['label']

        # Tokenize and pad sequences
        self.tokenizer.fit_on_texts(reviews)
        sequences = self.tokenizer.texts_to_sequences(reviews)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding=self.padding_type, truncating=self.trunc_type)

        return padded_sequences, np.array(labels)

    def build_model(self):
        self.model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            GlobalAveragePooling1D(),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    def save_model(self, filepath):
        # Save the model and tokenizer as a dictionary in .pkl format
        model_data = {
            "model": self.model,
            "tokenizer": self.tokenizer
        }
        joblib.dump(model_data, filepath)

    @staticmethod
    def main():
        # Load data
        train_data_dir = r"C:\Users\ASUS\Documents\MachineLearning\ML-Salmon\ML-Salmon\data\aclImdb\train"
        nn = NeuralNetwork()
        data = nn.load_data(train_data_dir)

        # Preprocess data
        X, y = nn.preprocess_data(data)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build, train, and save the model
        nn.build_model()
        nn.train(X_train, y_train, X_val, y_val, epochs=5)
        model_save_path = r"C:\Users\ASUS\Documents\MachineLearning\ML-Salmon\ML-Salmon\models_trained\trained_neural_network.pkl"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        nn.save_model(model_save_path)
        print(f"Model saved to {model_save_path}")

# Call main() outside the class definition
if __name__ == '__main__':
    NeuralNetwork.main()
