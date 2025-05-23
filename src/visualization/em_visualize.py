# visualize.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend Agg để tránh lỗi Tcl/Tk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Function to visualize classification report
def plot_classification_report(y_true, y_pred, file_path, model_name):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Plot the classification report as a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f', cbar=False)
    plt.title(f'Classification Report - {model_name}')
    plt.ylabel('Classes')
    plt.xlabel('Metrics')

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

# Function to visualize confusion matrix
def plot_confusion_matrix(y_true, y_pred, file_path, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

# Function to visualize ROC-AUC curve
def plot_roc_auc(y_true, y_scores, file_path, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])  # Lấy xác suất lớp Positive
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
    plt.legend(loc='lower right')

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

# Main function to execute visualizations for multiple models
def generate_visualizations(models, X_test, y_test, output_dir="models_trained/plots"):
    os.makedirs(output_dir, exist_ok=True)

    for model_name, model in models.items():
        print(f"Generating visualizations for {model_name}...")
        # Generate predictions
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        # Plot classification report
        classification_report_path = os.path.join(output_dir, f'classification_report_{model_name}.png')
        plot_classification_report(y_test, y_pred, classification_report_path, model_name)

        # Plot confusion matrix
        confusion_matrix_path = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
        plot_confusion_matrix(y_test, y_pred, confusion_matrix_path, model_name)

        # Plot ROC-AUC curve (if binary classification and model supports predict_proba)
        if y_scores is not None and len(np.unique(y_test)) == 2:
            roc_auc_path = os.path.join(output_dir, f'roc_auc_curve_{model_name}.png')
            plot_roc_auc(y_test, y_scores, roc_auc_path, model_name)

    print(f"Visualizations saved in {output_dir}")