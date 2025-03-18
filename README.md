# About
This repository is part of the Machine Learning Course (CO3117) at Ho Chi Minh University of Technology (HCMUT), containing the source code and the dataset used for both Assignment 1 and 2 of the course. 


# ML_Salmon group members
***Assignment 1 contribution:***
- Võ Trúc Sơn: Decision tree & Neutral Network implement
- Lê Văn Đức Anh: Bayesian Graphical implement
- Cao Quế Phương: Data preprocessing & HMM Graphical implement 
- Nguyễn Đức Hạnh Nhi: Decision tree & Naive Bayes implement
- Nông Thục Khuê: Data preprocessing & HMM Graphical implement

***Assignment 2 contribution:***

# Structure
```
project_root/
├── README.md
├── LICENSE
├── data/                                # Dữ liệu đầu vào
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
├── src/                                # Nguồn code
│   ├── models/                         # File mô hình
│   │   ├── predict_model.py            # File dự đoán
│   │   ├── train_model_bayesian.py     # File huấn luyện Bayesian
│   │   ├── train_model_decision.py     # File huấn luyện Decision Tree
│   │   └── train_model_neural.py       # File huấn luyện Neural Network
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

# Workflow
- Always activate the virtual environment before starting development.
- Update requirements.txt whenever new dependencies are added.
- Ensure consistent Python versions across the team.


## Follow these steps to contribute to the project:
- Fork the main repository on GitHub.
- Clone your fork:

```
git clone https://github.com/bubletofu/ML-Salmon.git
git remote add upstream https://github.com/bubletofu/ML-Salmon.git
```

- Create a feature branch:

```
git checkout -b feature/feature-name
```

- Keep your fork update:

```
git fetch upstream
git checkout develop
git merge upstream/develop
git push origin develop
```

- Develop feature branch:

```
git add .
git commit -m "description"
git push origin feature-name
```

- Rebase before submitting a Pull Request (PR):

```
git checkout feature-name
git fetch upstream
git rebase upstream/develop
git push -f origin feature-name
```

- Create a Pull Request from your feature branch to the upstream main branch.

# Key Points for Contributors
- Never commit directly to main or develop.
- Regularly sync your fork with the upstream repository.
- Rebase feature branches before submitting PRs.
- Use conventional and standard commit messages.


