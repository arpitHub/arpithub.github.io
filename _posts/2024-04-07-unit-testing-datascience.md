---
layout: post
title: Enhancing Data Science Workflow with Unit Testing
subtitle: Enhancing Data Science Workflow with Unit Testing
cover-img: /assets/img/pytest_1.jpg
thumbnail-img: /assets/img/pytest_thumb.png
share-img: /assets/img/pytest_2.jpg
gh-repo: arpithub/arpithub.github.io
gh-badge: [star, fork, follow]
tags: [datascience,testing,pytest,ml]
comments: true
---

In this blog post, we'll explore how to effectively use unit tests for a data science project using the Pytest framework. We'll use the popular Iris dataset as an example to demonstrate how to write tests for data preprocessing, model training, and evaluation functions.

#### What is Unit Testing?
Unit testing involves testing individual units (functions, methods, classes) of code to verify that they behave as intended. It allows you to detect and fix bugs early in the development process and ensures that changes to your code don't introduce unexpected issues.

#### Setting Up the Project
Before we dive into writing tests, let's set up our project structure:

```
data-science-project/
│
├── src/
│   ├── preprocessing.py
│   └── model.py
│── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
│
├── data/
│   └── iris.csv
│
└── requirements.txt
```

**src/:** Directory containing Python modules for data preprocessing (preprocessing.py) and model training (model.py).\
**tests/:** Directory for storing test modules (test_preprocessing.py and test_model.py).\
**data/:** Directory containing the Iris dataset (iris.csv).\
**requirements.txt:** File listing project dependencies (e.g., pytest, pandas, scikit-learn).\

#### Installing Dependencies
Make sure you have Python and pip installed. Create a conda environment and install the required packages:

```python
conda create --name=ml_test_project python=3.10
conda activate ml_test_project
pip install -r requirements.txt
```

**requirements.txt**
```python
pandas
pytest
scikit-learn
```

#### Writing Unit Tests with Pytest
Let's start by writing unit tests for our data preprocessing and modeling functions using Pytest.

1. Writing Tests for Data Preprocessing
Create `src/preprocessing.py` with a function to load the Iris dataset and preprocess it:

```python
import pandas as pd

def load_iris_dataset(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Drop rows with missing values
    df = df.dropna()
    # Encode target variable
    df['species'] = df['species'].astype('category').cat.codes
    return df
```

Now, create `tests/test_preprocessing.py` to write unit tests for the preprocessing functions:

```python
import pytest
from pathlib import Path
from src.preprocessing import load_iris_dataset, preprocess_data

data_dir = Path(__file__).parent.parent / 'data'  # Navigate up to the project root
iris_path = data_dir / 'iris.csv'


@pytest.fixture
def iris_data():
    return load_iris_dataset(iris_path)

def test_load_iris_dataset():
    df = load_iris_dataset(iris_path)
    assert not df.empty

def test_preprocess_data(iris_data):
    preprocessed_df = preprocess_data(iris_data)
    assert preprocessed_df.isna().sum().sum() == 0
    assert 'species' in preprocessed_df.columns

def test_missing_values():
    df = load_iris_dataset(iris_path)
    assert not df.isnull().values.any(), "Dataset contains missing values"

def test_no_duplicates():
    df = load_iris_dataset(iris_path)
    assert not df.duplicated().any(), "Dataset contains duplicate records"


def test_column_datatypes():
    df = load_iris_dataset(iris_path)
    expected_datatypes = {
        'sepal length (cm)': 'float64',
        'sepal width (cm)': 'float64',
        'petal length (cm)': 'float64',
        'petal width (cm)': 'float64',
        'species': 'category'
    }
    for col, dtype in expected_datatypes.items():
        assert df[col].dtype == dtype, f"Unexpected datatype for column {col}"

```

2. Writing Tests for Model Training and Evaluation
Create `src/model.py` with a function to train a simple classifier on the Iris dataset:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate_model(df):
    X = df.drop('species', axis=1)
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy
```

Now, create `tests/test_model.py` to write unit tests for the model training and evaluation function:

```python
import pytest
from pathlib import Path
from src.model import train_and_evaluate_model
from src.preprocessing import load_iris_dataset, preprocess_data
data_dir = Path(__file__).parent.parent / 'data'  # Navigate up to the project root
iris_path = data_dir / 'iris.csv'

@pytest.fixture
def preprocessed_iris_data():
    df = load_iris_dataset(iris_path)
    return preprocess_data(df)

def test_train_and_evaluate_model(preprocessed_iris_data):
    model, accuracy = train_and_evaluate_model(preprocessed_iris_data)
    assert accuracy > 0.8
```

#### Running the Tests
To run the tests using Pytest, navigate to the `tests` directory and execute:

```bash
pytest
```

#### Conclusion
By incorporating a suite of comprehensive unit tests, you can ensure the robustness and correctness of your data preprocessing, modeling, and evaluation workflows. Continuously expand and refine your tests to cover various scenarios and edge cases, enhancing the reliability and integrity of your data science projects.