import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Function to visualize classification report
def plot_classification_report(y_true, y_pred, file_path):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Plot the classification report as a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f', cbar=False)
    plt.title('Classification Report')
    plt.ylabel('Classes')
    plt.xlabel('Metrics')

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

# Function to visualize confusion matrix
def plot_confusion_matrix(y_true, y_pred, file_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

# Function to visualize ROC-AUC curve
def plot_roc_auc(y_true, y_scores, file_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

# Main function to execute visualizations
def generate_visualizations(model, X_test, test_labels, output_dir="models_trained/plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Generate the prediction results
    y_pred = model.predict(X_test)

    # Plot the classification report
    classification_report_path = os.path.join(output_dir, 'classification_report.png')
    plot_classification_report(test_labels, y_pred, classification_report_path)

    # Plot the confusion matrix
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(test_labels, y_pred, confusion_matrix_path)

    # Plot the ROC-AUC curve (if binary classification)
    if len(np.unique(test_labels)) == 2:
        y_scores = svm_model.decision_function(X_test)
        roc_auc_path = os.path.join(output_dir, 'roc_auc_curve.png')
        plot_roc_auc(test_labels, y_scores, roc_auc_path)

    print(f"Visualizations saved in {output_dir}")