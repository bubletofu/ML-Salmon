import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Function to plot sentiment distribution
def plot_sentiment_distribution(train_df):
    sentiment_counts = train_df['sentiment'].value_counts()
    plt.figure(figsize=(8,6))
    sentiment_counts.plot(kind='bar')
    plt.title('Sentiment Distribution in the Training Data')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Negative', 'Positive'], rotation=0)
    plt.show()

# Function to plot the top TF-IDF features
def plot_top_tfidf_features(X_train_tfidf, vectorizer):
    feature_names = np.array(vectorizer.get_feature_names())
    tfidf_scores = X_train_tfidf.sum(axis=0).A1
    sorted_idx = np.argsort(tfidf_scores)[::-1]

    plt.figure(figsize=(10,6))
    plt.barh(feature_names[sorted_idx][:10], tfidf_scores[sorted_idx][:10])
    plt.xlabel('TF-IDF Score')
    plt.title('Top 10 TF-IDF Features')
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

# Function to plot learning curves
def plot_learning_curves(training_accuracies, validation_accuracies):
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.title('Learning Curves')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
