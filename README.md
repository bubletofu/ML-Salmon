# About
This repository is part of the Machine Learning Course (CO3117) at Ho Chi Minh University of Technology (HCMUT), containing the source code and the dataset used for both Assignment 1 and 2 of the course. 


## ML_Salmon group members
***Assignment 1 contribution:***
- Võ Trúc Sơn: Decision tree & Neutral Network implement
- Lê Văn Đức Anh: Bayesian Graphical implement & Naive Bayes implement
- Cao Quế Phương: Data preprocessing & HMM Graphical implement 
- Nguyễn Đức Hạnh Nhi: Decision tree & Naive Bayes implement
- Nông Thục Khuê: Data preprocessing & HMM Graphical implement

***Assignment 2 contribution:***

# Structure
```
project_root/
├── README.md
├── LICENSE
├── data/                               # Dữ liệu đầu vào
│   ├── aclImdb/                        # Tập dữ liệu IMDb
│   │   ├── train/                      # Dữ liệu huấn luyện
│   │   │   ├── neg/                    # Đánh giá tiêu cực
│   │   │   └── pos/                    # Đánh giá tích cực
│   │   └── test/                       # Dữ liệu kiểm tra
│   │       ├── neg/                    # Đánh giá tiêu cực
│   │       └── pos/                    # Đánh giá tích cực
├── models_trained/                     # Mô hình đã huấn luyện
│   ├── trained_bayesian_network.pkl    # Mô hình Bayesian Network đã lưu
│   ├── trained_decision_tree.pkl       # Mô hình Decision Tree đã lưu
│   └── trained_neural_network.pkl      # Mô hình Neural Network đã lưu
│   └── trained_naive_bayyesian.pkl     # Mô hình Naive Bayes đã lưu
│   └── trained_hmm_model.pkl           # Mô hình HMM đã lưu
├── src/                                # Nguồn code
│   ├── models/                         # File mô hình
│   │   ├── predict_model.py            # File dự đoán
│   │   ├── predict.py                  # File show metrics của model
│   │   ├── BayesianNetwork.py          # File train mô hình Bayesian Network
│   │   ├── dT.py                       # File train mô hình decision tree
│   │   ├── NN.py                       # File train mô hình neural network
│   │   ├── HMM.py                      # File train mô hình HMM
│   │   ├── dT.py                       # File train mô hình decision tree
│   │   ├── NaiveBayes.py               # File train mô hình Naive Bayes
│   ├── __init__.py                     # Khởi tạo package
├── tests/                              # Unit tests
│   ├── __init__.py                     # Khởi tạo module test
│   └── test_models.py                  # File test cho các mô hình
├── requirements.txt                    # Danh sách thư viện phụ thuộc
└── .gitignore                          # File bỏ qua cho Git

```

Where:
- `notebooks/`: Jupyter notebooks for assignments and experiments.
- `data/`: Datasets used in the course.
- `src/`: Source code for data processing, feature engineering, and model training.
- `models/`: Saved models and experimental artifacts.
- `reports/`: Analysis reports and final project documentation.
- `tests/`: Unit tests for code validation.

# Setup
**1. Environment requirement:**
- Python version: at least 3.9 (Recommended: 3.11)
- Supported operating systems:
  - Window 10/11
  - MacOS
  - Linux
##
**2. Setup steps:**
- For Windows:
  - Install Python from the official website.
  - Open Command Prompt and run the following commands:


```
# Create virtual environment
python -m venv venv
# Activate virtual environment
.\venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
```


- For MacOS/Linux:
  - Install Python using Homebrew (macOS) or your package manager (Linux).
  - Run the following commands:

``` 
# Create virtual environment
python3 -m venv venv
# Activate virtual environment
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

# Troubleshooting
- Common Dependency Issues
  - Ensure all team members use matching Python versions.
  - Use pip freeze > requirements.txt to capture exact dependency versions.
  - Check for architecture compatibility (x86/ARM).

# Potential Resolution Steps
- Update pip:
```
python -m pip install --upgrade pip
```

- Install build tools:
  - Windows: Install Visual C++ Build Tools.
  - macOS: Install Xcode Command Line Tools.
  - Linux: Install the build-essential package.


# *Follow these steps to contribute to the project:*
- Fork the main repository on GitHub.
- Clone your fork:

```
git clone https://github.com/bubletofu/ML-Salmon.git
```

- Create a origin branch:

```
git checkout branchName
```

- Keep your fork update:

```
git pull origin branchName
git checkout main
git merge branchName
git pull origin main
git add .
git commit -m "Merge from branchName"
git push origin main
```

- Develop origin branch:

```
git add .
git commit -m "description"
git push origin branchName
```

- Pull request branch:

```
git checkout branchName
git pull origin branchName
```


# *Key Points for Contributors*
- Only leader commit to main, member please create branch for committion
- Remember to add commit message for easily following future development
- Remember to checkout your branch and pull before code
- Use conventional and standard commit messages.

# Model Overview
## Decision tree
### Overview of Decision Tree  

#### What is a Decision Tree?  

A Decision Tree is a machine learning model used for solving classification and regression problems. It works like a tree diagram, where each **node** represents a question or decision, each **branch** represents a choice, and each **leaf** represents the final outcome (a label or predicted value). Imagine playing a game of "Guess the Object" with your friends: you ask questions like, *"Is it an animal?"* and then *"Does it have fur?"* to narrow down the answer—Decision Trees operate in a similar way!  

#### How Decision Trees Work  

1. **Building the Tree**:  
   - The model starts with a **root node**, containing all the data.  
   - It selects a feature and a threshold to split the data into two or more groups. For example, if classifying movies as *good* or *bad* based on length, a question could be: *"Is the movie longer than 120 minutes?"*  
   - The feature selection is based on criteria such as **Gini Index** (measuring group purity) or **Information Gain** (measuring the reduction in uncertainty after splitting).  
   - This process repeats for each subgroup, generating new nodes and branches until:  
     - All data in a group belong to the same class (pure leaves).  
     - A maximum tree depth is reached, or no more useful features remain for splitting.  

2. **Making Predictions**:  
   - For new data, the model starts at the root node, follows branches based on feature values, and reaches a leaf node to make a prediction.  
   - Example: If a movie is 150 minutes long (greater than 120 minutes) and was produced after 2000, the tree might predict it as *good*.  

#### Advantages of Decision Trees  

- **Easy to understand and visualize**: Decision Trees resemble flowcharts, making them intuitive even for non-experts.  
- **No need for data normalization**: Unlike some models, Decision Trees don’t require feature scaling.  
- **Handles both numerical and categorical data**: Works well with features like movie length (numerical) or genre (categorical).  

#### Disadvantages of Decision Trees  

- **Prone to overfitting**: A deep tree may learn noise from the training data, leading to poor generalization.  
- **Sensitive to data changes**: A small modification in data can alter the tree’s structure significantly.  
- **Not always the best for complex problems**: Compared to models like Neural Networks, Decision Trees may perform worse on intricate tasks like text classification.  

#### Real-World Applications  

- **Email classification**: Identifying whether an email is *spam* or *not spam* based on features like keywords and email length.  
- **Medical diagnosis**: Predicting whether a patient has a disease based on symptoms (e.g., fever, cough).  
- **Customer segmentation**: Determining whether a customer will buy a product based on age, income, and preferences.  

#### Simple Example  

Suppose we want to predict whether a student will *pass* or *fail* an exam based on study hours and test scores:  
- Data: [5 hours, 8 score, Pass], [2 hours, 4 score, Fail], [6 hours, 7 score, Pass].  
- The tree might be structured like this:  
  - Root node: *"Study hours > 3?"*  
    - If *Yes* → *"Score > 6?"* → Pass.  
    - If *No* → Fail.  
- Prediction: A student who studies for 4 hours and scores 9 → *Pass*.  

#### Improving Decision Trees  

- **Limit tree depth (`max_depth`)**: Helps prevent overfitting.  
- **Use Random Forest**: Combining multiple Decision Trees improves accuracy.  
- **Preprocess data**: Removing noise and selecting better features enhances performance.  

### Conclusion  

Decision Trees are simple yet powerful tools in machine learning, especially when interpretability is important. While they have limitations, they serve as the foundation for advanced models like Random Forest and Gradient Boosting. If you're new to machine learning, Decision Trees are a great starting point to understand how models make "decisions" based on data!
## Neural Network
## Naive Bayesian
## Bayesian Network
## HMM
