#!/bin/bash

mkdir -p ml-course/{notebooks/{assignment1,assignment2,final_project},data/{raw,processed,external},src/{data,features,models,visualization},models/{trained,experiments},reports/{figures,final_project},tests}
touch ml-course/{README.md,requirements.txt,.gitignore}
touch ml-course/notebooks/assignment1/{exploration.ipynb,submission.ipynb}
touch ml-course/src/{__init__.py,data/{__init__.py,make_dataset.py,preprocess.py},features/{__init__.py,build_features.py},models/{__init__.py,train_model.py,predict_model.py},visualization/{__init__.py,visualize.py}}
touch ml-course/tests/{__init__.py,test_data.py,test_models.py}
