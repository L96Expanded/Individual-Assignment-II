# Machine Learning Foundations – Bike Sharing Prediction Assignment II

## Overview

This repository contains the implementation for Assignment II in the "Machine Learning Foundations" course. In this assignment, we predict the hourly bike rental counts (`cnt`) using the UCI Bike Sharing Dataset. The modeling process includes exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, and final model evaluation. We build and compare three regression models:
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor (using XGBoost)

## Repository Structure

- **`assignment2_YourName.ipynb`**  
  The main Jupyter Notebook that includes all code, visualizations, and explanations required for the assignment. It covers every step from data loading and preprocessing through to model evaluation.
  
- **`ML_assignment_2-3.pdf`**  
  The assignment brief and requirements document.
  
- **`hour.csv`**  
  The Bike Sharing Dataset file (use the `hour.csv` file from the UCI Machine Learning Repository).  
  *Note: If this file is not included in the repository, please download it from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).*
  
- **`README.md`**  
  This file, which provides an overview and instructions for the project.

## Prerequisites

To run the notebook, ensure you have Python 3 and the following packages installed:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Scikit-optimize (for BayesSearchCV)

You can install the required packages using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost scikit-optimize
```

## Assignment Workflow

### 1. Exploratory Data Analysis (EDA)
- **Load the dataset:** Read the `hour.csv` file.
- **Analyze target variable (`cnt`):** Check distribution and skewness.
- **Investigate features:** Visualize how variables (such as hour, weekday, month, season, weather-related features) affect bike rentals.
- **Feature cleaning:** Drop the redundant columns (e.g., `instant`, `dteday`, `casual`, and `registered`).

### 2. Data Splitting
- Split the dataset into training (60%), validation (20%), and test (20%) sets.
- Ensure splitting is performed prior to any feature engineering to avoid data leakage.

### 3. Feature Engineering
- **Cyclical Encoding:** Transform cyclical features (e.g., `hr` and `weekday`) using sine and cosine functions.
- **One-hot Encoding:** Encode categorical variables such as `season`, `weathersit`, and `mnth`.
- **Scaling:** Standardize continuous features (`temp`, `atemp`, `hum`, and `windspeed`).
- **Interaction Terms:** Optionally create interaction features (e.g., `temp * hum`) if justified by the EDA.

### 4. Model Training
- **Linear Regression:** Serve as a baseline model with evaluation metrics—MSE, MAE, and R².
- **Random Forest Regressor:** Evaluate model performance with default settings and analyze feature importance.
- **Gradient Boosting Regressor (XGBoost):** Train and evaluate with initial parameters.

### 5. Hyperparameter Tuning
- **Random Forest:** Use RandomizedSearchCV to tune parameters like `n_estimators`, `max_depth`, etc.
- **XGBoost:** Use BayesSearchCV for tuning parameters such as `learning_rate`, `n_estimators`, `max_depth`, and `subsample`.

### 6. Final Model Selection and Testing
- Select the best performing model based on validation metrics.
- Retrain on combined training and validation sets.
- Evaluate the final model on the test set and report the metrics.

## Running the Notebook

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   ```
2. **Navigate to the repository directory:**
   ```bash
   cd <repository_directory>
   ```
3. **Launch Jupyter Notebook or JupyterLab:**
   ```bash
   jupyter notebook assignment2_YourName.ipynb
   ```
4. **Run all cells** in sequential order to perform the analysis and see the outputs.

## Additional Notes

- **Consistency in Features:**  
  The notebook handles the issue of mismatched feature names (e.g., unseen dummy columns like `weathersit_4`) by aligning the columns in the training, validation, and test sets.
  
- **Iterative Development:**  
  The notebook embodies an iterative process where you may revisit EDA and feature engineering based on model performance.
  
- **Documentation:**  
  Detailed markdown cells explain the reasoning behind every transformation, model selection, and evaluation process.
