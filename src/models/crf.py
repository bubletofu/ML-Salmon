import os
import joblib
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn_crfsuite import CRF

# If using NLTK tokenization, download required resources
nltk.download('punkt')

# Define base data path
BASE_DATA_PATH = "data/aclImdb"


def load_data(data_path):
    """
    Load text data from positive and negative subdirectories.
    Returns tokenized sentences and labels per token.
    For sequence labeling, labels per token are the same as review label.
    """
    texts, labels = [], []
    label_map = {"pos": "POS", "neg": "NEG"}
    for subdir in ["pos", "neg"]:
        path = os.path.join(data_path, subdir)
        for fname in os.listdir(path):
            with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                text = f.read().strip()
            tokens = nltk.word_tokenize(text)
            texts.append(tokens)
            # assign same sentiment tag for each token
            labels.append([label_map[subdir]] * len(tokens))
    return texts, labels


def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
    }
    if i > 0:
        prev = sent[i-1]
        features.update({
            '-1:word.lower': prev.lower(),
            '-1:word.istitle': prev.istitle(),
            '-1:word.isupper': prev.isupper(),
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        nxt = sent[i+1]
        features.update({
            '+1:word.lower': nxt.lower(),
            '+1:word.istitle': nxt.istitle(),
            '+1:word.isupper': nxt.isupper(),
        })
    else:
        features['EOS'] = True
    return features


def extract_features(texts):
    return [[word2features(s, i) for i in range(len(s))] for s in texts]


if __name__ == '__main__':
    # Load and split data
    texts, labels = load_data(os.path.join(BASE_DATA_PATH, 'train'))
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42)

    # Feature extraction
    X_train_feats = extract_features(X_train)
    X_val_feats = extract_features(X_val)

    # Initialize and train CRF
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train_feats, y_train)

    # Save model
    os.makedirs('models_trained', exist_ok=True)
    joblib.dump(crf, 'models_trained/crf_sentiment.pkl')

    # Evaluate
    y_pred = crf.predict(X_val_feats)
    # Flatten for classification report
    y_val_flat = [label for seq in y_val for label in seq]
    y_pred_flat = [label for seq in y_pred for label in seq]
    print('Validation Report:')
    print(classification_report(y_val_flat, y_pred_flat))
    print('Confusion Matrix:')
    print(confusion_matrix(y_val_flat, y_pred_flat))

        # Hàm predict sentiment
    def predict_sentiment(texts, model_path='models_trained/crf_sentiment.pkl'):
        crf_loaded = joblib.load(model_path)
        tokenized = [nltk.word_tokenize(t) for t in texts]
        feats = extract_features(tokenized)
        preds = crf_loaded.predict(feats)
        results = []
        for seq in preds:
            seq_list = list(seq)
            majority = max(set(seq_list), key=seq_list.count)
            results.append('Positive' if majority == 'POS' else 'Negative')
        return results

    # Ví dụ
    sample = ["This movie was fantastic with great performances."]
    print(predict_sentiment(sample))