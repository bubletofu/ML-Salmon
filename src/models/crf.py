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
    print(predict_sentiment(sample))yyrrr4f

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

    def plot_classification_report_bin(y_true, y_pred, file_path):
        report = classification_report(y_true, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        plt.figure(figsize=(8,6))
        plt.imshow(df.iloc[:-1, :-1], aspect='auto', interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(df.columns)-1), df.columns[:-1], rotation=45)
        plt.yticks(range(len(df.index)-1), df.index[:-1])
        plt.title('Classification Report')
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()

    def plot_confusion_matrix_bin(y_true, y_pred, file_path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        for i in (0,1):
            for j in (0,1):
                plt.text(j, i, cm[i,j], 
                        ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black')
        plt.xticks([0,1], ['NEG','POS'])
        plt.yticks([0,1], ['NEG','POS'])
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()

    def plot_roc_auc_bin(y_true, y_score, file_path):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0,1], [0,1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()

    # 2) Hàm tổng hợp
    def generate_visualizations_inline(model, X, y_seq, output_dir="models_trained/plots"):
        os.makedirs(output_dir, exist_ok=True)

        if any(isinstance(el, list) for el in y_seq):
            y_seq = ['POS' if seq.count('POS') > seq.count('NEG') else 'NEG'
                    for seq in y_seq]
        
        label_bin = np.array([1 if lab=='POS' else 0 for lab in y_seq])
        
        y_pred_seq = model.predict(X)
        y_pred = np.array([1 if lab=='POS' else 0 for lab in y_pred_seq])
        try:
            y_score = model.decision_function(X)
        except AttributeError:
            y_score = np.array([pred_seq.count('POS')/len(pred_seq) 
                                for pred_seq in model.crf.predict(X)])
        
        plot_classification_report_bin(label_bin, y_pred, 
                                    os.path.join(output_dir, 'classification_report.png'))
        plot_confusion_matrix_bin(label_bin, y_pred, 
                                os.path.join(output_dir, 'confusion_matrix.png'))
        plot_roc_auc_bin(label_bin, y_score, 
                        os.path.join(output_dir, 'roc_auc_curve.png'))
        
        print(f"✅ Saved plots to {output_dir}")

    os.makedirs("models_trained/plots/DM", exist_ok=True)
    generate_visualizations_inline(wrapped_crf, X_val_feats, y_val, 
                                output_dir="models_trained/plots/DM")
