## About
This repository is part of the Machine Learning Course (CO3117) at Ho Chi Minh University of Technology (HCMUT). It serves as a centralized resource for students to access course materials, homework solutions, code, and project documentation.


# ML member
- Võ Trúc Sơn: Decision tree & Neutral Network implement
- Lê Văn Đức Anh: Bayesian Graphical implement
- Cao Quế Phương: Data preprocessing & HMM Graphical implement 
- Nguyễn Đức Hạnh Nhi: Decision tree & Naive Bayes implement
- Nông Thục Khuê: Data preprocessing & HMM Graphical implement

# ML Course

This repository contains materials, code, and notebooks for an ML course.

## Structure
- project_root/
  - README.md
  - LICENSE
  - requirements.txt
  - dataset/                              # Data storage
  - models/                              # Model artifacts
  - notebooks/                           # Jupyter notebooks
  - reports/
  - src/                                 # Source code
      - __init__.py
      - config.py
      - data/
          - __init__.py
          - make_dataset.py
          - preprocess.py
          - features/
          - __init__.py
          - build_features.py
      - models/
          - __init__.py
          - predict_model.py
          - train_model.py
      - visualization/
          - __init__.py
          - visualize.py
  - tests/                               # Unit tests
      - __init__.py
      - test_data.py
      - test_models.py

Where:
- `notebooks/`: Jupyter notebooks for assignments and experiments.
- `data/`: Datasets used in the course.
- `src/`: Source code for data processing, feature engineering, and model training.
- `models/`: Saved models and experimental artifacts.
- `reports/`: Analysis reports and final project documentation.
- `tests/`: Unit tests for code validation.

## Setup
1. Environment requirement:
- Python version: at least 3.9 (Recommended: 3.11)
- Supported operating systems:
+ Window 10/11
+ MacOS
+ Linux
2. Setup steps:
- For Windows:
+ Install Python from the official website.
+ Open Command Prompt and run the following commands:


```
# Create virtual environment
python -m venv venv
# Activate virtual environment
.\venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
```


- For MacOS/Linux:
+ Install Python using Homebrew (macOS) or your package manager (Linux).
+ Run the following commands:

``` 
# Create virtual environment
python3 -m venv venv
# Activate virtual environment
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

## Troubleshooting
- Common Dependency Issues
+ Ensure all team members use matching Python versions.
+ Use pip freeze > requirements.txt to capture exact dependency versions.
+ Check for architecture compatibility (x86/ARM).

## Potential Resolution Steps
- Update pip:
```
python -m pip install --upgrade pip
```

- Install build tools:
+ Windows: Install Visual C++ Build Tools.
+ macOS: Install Xcode Command Line Tools.
+ Linux: Install the build-essential package.

## Workflow
- Always activate the virtual environment before starting development.
- Update requirements.txt whenever new dependencies are added.
- Ensure consistent Python versions across the team.

## Contributing
# Follow these steps to contribute to the project:
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

## Key Points for Contributors
- Never commit directly to main or develop.
- Regularly sync your fork with the upstream repository.
- Rebase feature branches before submitting PRs.
- Use conventional and standard commit messages.


