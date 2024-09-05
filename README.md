# Premier League Prediction: An Analytical Study Using Machine Learning

## Project Overview

This project aims to predict match outcomes (Home Win, Draw, or Away Win) for the Premier League using various machine learning techniques. The analysis includes the implementation of models such as Random Forest, Gradient Boosting, and Support Vector Machine (SVM) to evaluate key football match metrics including possession, shots, goals, passes, and more.

## Purpose

The primary goal of this project is to develop, train, and test machine learning models on historical Premier League match data. This helps in predicting future match outcomes and extracting actionable insights based on match statistics, which can be valuable for teams, analysts, and fans alike.

## Input Format

The input dataset should be a CSV file containing the following match statistics:
- Team names
- Possession percentages
- Shots (on target, off target)
- Goals scored
- Tackles, fouls, and other relevant football statistics

## Output Format

The output from the model includes:
- **Predicted match outcome**: Home Win, Draw, or Away Win
- **Performance metrics**: Accuracy, Precision, Recall, F1-score
- **Confusion matrix** and **classification report** for each model

## Code Overview

### 1. Data Preprocessing

- **Input**: Raw dataset in CSV format with football match statistics.
- **Output**: Cleaned and transformed dataset ready for model training.
- **Functions used**: `df.rename()`, `df.isnull()`, `df.describe()`, `df.info()`

### 2. Feature Selection and Label Encoding

- **Input**: Cleaned dataset.
- **Output**: Features (X) and target variable (y) for training.
- **Function used**: `LabelEncoder()` for encoding categorical target values (Match Outcome).

### 3. Model Building and Training

- **Random Forest Classifier**:
  - **Model**: `RandomForestClassifier()`
  - **Hyperparameter tuning**: Using `GridSearchCV`
  - **Parameters**: `n_estimators`, `max_depth`

- **Gradient Boosting Classifier**:
  - **Model**: `GradientBoostingClassifier()`
  - **Hyperparameter tuning**: Using `GridSearchCV`
  - **Parameters**: `n_estimators`, `learning_rate`, `max_depth`

- **Support Vector Machine (SVM)**:
  - **Model**: `SVC()`
  - **Hyperparameter tuning**: Using `GridSearchCV`
  - **Parameters**: `C`, `kernel`

### 4. Model Evaluation

- **Output**: Evaluation metrics for all models including accuracy, precision, recall, F1-score.
- **Function**: `evaluate_model(predictions, y_test, model_name)` to display the evaluation metrics.

## Key Parameters and Functions

### GridSearchCV

- **Purpose**: To perform hyperparameter tuning.
- **Key Parameters**:
  - `param_grid`: Dictionary of hyperparameters for each model.
  - `cv`: Number of folds for cross-validation (set to 5).

### SMOTE (Synthetic Minority Over-sampling Technique)

- **Purpose**: Balances the dataset by oversampling the minority class.
- **Function**: `smote.fit_resample(X_train, y_train)` to generate balanced datasets.

### Pipeline

- **Purpose**: Standardizes preprocessing and training steps for each model.
- **Key Components**:
  - `StandardScaler()`: Normalizes the features.
  - Classifier models (RandomForest, GradientBoosting, or SVM).

## Outputs

- **Accuracy**: Measures how accurate the model is at predicting outcomes.
- **Precision**: The ratio of correctly predicted positive observations.
- **Recall**: The ratio of correctly predicted actual positives.
- **F1 Score**: Weighted average of precision and recall.

