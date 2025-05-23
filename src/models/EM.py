# EM.py
import os
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import joblib  # Để lưu và tải mô hình/TF-IDF
from glob import glob  # Để đọc file nhanh hơn
import time

# Hàm đọc dữ liệu nhanh hơn với glob
def load_imdb_data(data_dir, max_samples_per_class=None):
    print(f"Loading data from {data_dir}...")
    start_time = time.time()
    data = []
    labels = []
    
    # Đọc file từ thư mục pos
    pos_files = glob(os.path.join(data_dir, 'pos', '*.txt'))
    if max_samples_per_class:
        pos_files = pos_files[:max_samples_per_class]
    for file in pos_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data.append(f.read())
                labels.append(1)
        except UnicodeDecodeError:
            print(f"Error reading {file}, trying latin-1 encoding")
            with open(file, 'r', encoding='latin-1') as f:
                data.append(f.read())
                labels.append(1)
    
    print(f"Loaded {len(data)} positive samples in {time.time() - start_time:.2f} seconds")
    
    # Đọc file từ thư mục neg
    neg_files = glob(os.path.join(data_dir, 'neg', '*.txt'))
    if max_samples_per_class:
        neg_files = neg_files[:max_samples_per_class]
    for file in neg_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data.append(f.read())
                labels.append(0)
        except UnicodeDecodeError:
            print(f"Error reading {file}, trying latin-1 encoding")
            with open(file, 'r', encoding='latin-1') as f:
                data.append(f.read())
                labels.append(0)
    
    print(f"Loaded {len(data)} total samples from {data_dir} in {time.time() - start_time:.2f} seconds")
    return data, labels

# Đường dẫn dữ liệu
train_dir = r'C:\Users\ASUS\Documents\MachineLearning\ML-Salmon\ML-Salmon\data\aclImdb\train'
test_dir = r'C:\Users\ASUS\Documents\MachineLearning\ML-Salmon\ML-Salmon\data\aclImdb\test'

# Đọc dữ liệu (không giới hạn mẫu để dùng toàn bộ dataset)
print("Loading training data...")
X_train, y_train = load_imdb_data(train_dir)  # Dùng toàn bộ 25,000 mẫu
print("Loading test data...")
X_test, y_test = load_imdb_data(test_dir)    # Dùng toàn bộ 25,000 mẫu

# Tiền xử lý dữ liệu và lưu ma trận TF-IDF
tfidf_file = 'tfidf_matrices.pkl'
if os.path.exists(tfidf_file):
    print("Loading precomputed TF-IDF matrices...")
    with open(tfidf_file, 'rb') as f:
        X_train_tfidf, X_test_tfidf, vectorizer = pickle.load(f)
else:
    print("Preprocessing data with TF-IDF...")
    start_time = time.time()
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF preprocessing done in {time.time() - start_time:.2f} seconds")
    
    # Lưu ma trận TF-IDF để tái sử dụng
    with open(tfidf_file, 'wb') as f:
        pickle.dump((X_train_tfidf, X_test_tfidf, vectorizer), f)

# Base models (tối ưu hóa để tăng tốc)
base_models = [
    ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),  # Giới hạn độ sâu cây
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('nb', MultinomialNB())
]

# Bagging with Random Forest
print("Training Random Forest...")
start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)
print(f"Random Forest training completed in {time.time() - start_time:.2f} seconds")
joblib.dump(rf_model, 'rf_model.pkl')  # Lưu mô hình

# Boosting with AdaBoost
print("Training AdaBoost...")
start_time = time.time()
ada_model = AdaBoostClassifier(n_estimators=20, random_state=42)
ada_model.fit(X_train_tfidf, y_train)
y_pred_ada = ada_model.predict(X_test_tfidf)
print(f"AdaBoost training completed in {time.time() - start_time:.2f} seconds")
joblib.dump(ada_model, 'ada_model.pkl')

# Voting Classifier
print("Training Voting Classifier...")
start_time = time.time()
voting_clf = VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)
voting_clf.fit(X_train_tfidf, y_train)
y_pred_voting = voting_clf.predict(X_test_tfidf)
print(f"Voting Classifier training completed in {time.time() - start_time:.2f} seconds")
joblib.dump(voting_clf, 'voting_model.pkl')

# Stacking Classifier (chạy riêng nếu cần vì chậm)
print("Training Stacking Classifier...")
start_time = time.time()
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(max_iter=1000), n_jobs=-1)
stacking_clf.fit(X_train_tfidf, y_train)
y_pred_stack = stacking_clf.predict(X_test_tfidf)
print(f"Stacking Classifier training completed in {time.time() - start_time:.2f} seconds")
joblib.dump(stacking_clf, 'stacking_model.pkl')

# Evaluate models
print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

print("\nAdaBoost Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_ada))
print(classification_report(y_test, y_pred_ada))

print("\nVoting Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_voting))
print(classification_report(y_test, y_pred_voting))

print("\nStacking Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_stack))
print(classification_report(y_test, y_pred_stack))

# Cross-validation for Random Forest (chỉ chạy nếu cần, vì chậm)
print("\nPerforming cross-validation for Random Forest...")
start_time = time.time()
cv_scores_rf = cross_val_score(rf_model, X_train_tfidf, y_train, cv=3, n_jobs=-1)
print(f"Random Forest Cross-validation scores: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")
print(f"Cross-validation completed in {time.time() - start_time:.2f} seconds")

# Plot model comparison
accuracies = [
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_ada),
    accuracy_score(y_test, y_pred_voting),
    accuracy_score(y_test, y_pred_stack)
]
models = ['Random Forest', 'AdaBoost', 'Voting', 'Stacking']
plt.bar(models, accuracies, color=['#4CAF50', '#2196F3', '#FFC107', '#FF5722'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.ylim(0, 1)
plt.savefig('model_comparison.png')
plt.show()