# Bank Customer Churn Prediction

A supervised machine learning project to identify customers at risk of churning, using Logistic Regression and KNN — with SMOTE for class imbalance handling and GridSearchCV for hyperparameter tuning.

---

## Overview

Customer churn is one of the most costly challenges in banking. This project builds an end-to-end binary classification pipeline on 10,000 bank customer records to predict whether a customer will exit — enabling data-driven, proactive retention strategies.

---

## Dataset

**Source:** [Kaggle – Churn Modelling Dataset]([https://www.kaggle.com/datasets/shubh0799/churn-modelling](https://raw.githubusercontent.com/SagarChhabriya/data-science/refs/heads/main/datasets/04-Binary-Classification/bank_churn_modelling.csv?utm_source=chatgpt.com))  

**Size:** 10,000 records × 13 features | No nulls | No duplicates

| Feature | Type | Description |
|---|---|---|
| `CreditScore` | Integer | Customer's credit score |
| `Geography` | Categorical | France, Germany, or Spain |
| `Gender` | Categorical | Male / Female |
| `Age` | Integer | Customer age in years |
| `Tenure` | Integer | Years with the bank |
| `Balance` | Float | Account balance (EUR) |
| `Num_Of_Products` | Integer | Number of bank products held |
| `Has_Credit_Card` | Binary | Holds a credit card (1 = Yes) |
| `Is_Active_Member` | Binary | Active member status (1 = Yes) |
| `Estimated_Salary` | Float | Estimated annual salary (EUR) |
| `Churn` | Binary | **Target** — 1 = Churned, 0 = Retained |

> **Note:** The dataset is imbalanced (~80% retained, ~20% churned). SMOTE was applied to the training set to balance the classes before model training.

---

## Project Pipeline

1. **Data Loading**
   - Load dataset from raw GitHub URL

2. **Exploratory Data Analysis (EDA)**
   - Statistical summary of numerical and categorical features
   - Univariate analysis — gender, geography, churn distribution, age
   - Bivariate analysis — churn by gender, tenure, geography, age group, activity

3. **Preprocessing**
   - Drop irrelevant columns (`CustomerId`, `Surname`)
   - One-Hot Encoding for `Geography` and `Gender`
   - Train-Test Split — 80/20 (stratified)
   - Feature Scaling — `StandardScaler` fitted on training data only

4. **Baseline Modelling (Before SMOTE)**
   - Logistic Regression (unscaled)
   - KNN k=5 (unscaled)

5. **Class Imbalance Handling — SMOTE**
   - Training set balanced: 6,370 → 12,740 samples

6. **Model Training (After SMOTE)**
   - Logistic Regression
   - KNN

7. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - ROC-AUC Score and ROC Curve
   - 5-Fold Cross-Validation

8. **Hyperparameter Tuning**
   - KNN → K values tested: 1–20
   - Logistic Regression → GridSearchCV on `C` → Best C = 2

9. **Final Model — Tuned Logistic Regression (C=2)**
   - Train final model using best hyperparameter from GridSearchCV

---

## Model Results

### Baseline (Before SMOTE)

| Model | Accuracy | Recall (Churn) | F1 (Churn) |
|---|---|---|---|
| Logistic Regression | 80.75% | 0.18 | 0.28 |
| KNN (k=5) | 76.40% | 0.08 | 0.13 |

### After SMOTE

| Model | Accuracy | Recall (Churn) | F1 (Churn) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 71.55% | 0.70 | 0.50 | 0.777 |
| KNN | 72.45% | 0.65 | 0.49 | 0.755 |

### Cross-Validation (After SMOTE)

| Model | CV Score |
|---|---|
| Logistic Regression | 0.713 |
| KNN | 0.846 |

> **Note:** After SMOTE, overall accuracy decreases slightly — this is expected. The model shifts from predicting only the majority class to learning both classes equally. **Recall and F1-Score are the primary metrics** for this problem, since missing a churner costs more than a false alarm.

---

## Key Findings

| Factor | Insight |
|---|---|
| Geography | Germany has a 32% churn rate — nearly 2× France (16%) and Spain (17%) |
| Age | Senior customers (45–60) churn at 51% — the highest of any age group |
| Activity | Inactive members churn at 27% vs 14% for active members |
| Products | Customers with 3–4 products churn at disproportionately high rates |
| Balance | Very high or zero balances correlate with elevated churn risk |
| Top Predictor | Age has the highest positive coefficient (0.87) in Logistic Regression |

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- Jupyter Notebook or JupyterLab

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/kainat-fareed/bank-churn-prediction.git
cd bank-churn-prediction

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn

# 3. Launch the notebook
jupyter notebook bank-churn-prediction.ipynb
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `Pandas` | Data loading and manipulation |
| `NumPy` | Numerical computation |
| `Matplotlib` & `Seaborn` | Data visualisation |
| `Scikit-learn` | ML models, scaling, evaluation, GridSearchCV |
| `imbalanced-learn` | SMOTE for class imbalance handling |
