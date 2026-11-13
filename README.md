# Customer Churn Prediction

## Project Overview

This project focuses on predicting customer churn using Machine Learning.

Customer churn occurs when customers stop using a company's products or services. The goal of this project is to build a classification model that identifies customers who are likely to leave, helping businesses understand churn patterns and take preventive actions.

The project uses the **IBM Telco Customer Churn dataset** and explores different classification techniques, with a focus on XGBoost.

## Dataset

- **Dataset:** IBM Telco Customer Churn
- **Rows:** 7,043
- **Original columns:** 21
- **Target variable:** `Churn`
- **Problem type:** Binary Classification

The dataset contains customer demographics, subscribed services, account information, contract types, payment methods, and billing details.

Dataset source: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Project Workflow

### 1. Data Cleaning and Preprocessing

- Checked missing values and duplicate records.
- Converted `TotalCharges` from object to numeric.
- Handled 11 missing `TotalCharges` values by filling them with 0 after checking that the corresponding customers had zero tenure.
- Encoded binary Yes/No variables.
- Applied One-Hot Encoding to categorical features.
- Removed `customerID` and target-derived columns from the input features.
- Split the dataset into training and testing sets using stratification.

### 2. Exploratory Data Analysis (EDA)

Explored customer characteristics and their relationships with churn using Pandas, Matplotlib, and Seaborn.

Important findings:

- Customers with month-to-month contracts had a churn rate of approximately 42.71%.
- Customers with two-year contracts had a much lower churn rate of approximately 2.83%.
- Fiber optic internet customers had a churn rate of approximately 41.89%.
- Electronic check users had a churn rate of approximately 45.29%.
- Customers who churned generally had shorter tenure and higher monthly charges.

### 3. Model Training

Two classification algorithms were tested:

- Random Forest Classifier
- XGBoost Classifier

Class imbalance was addressed using class weighting:

- `class_weight='balanced'` for Random Forest.
- `scale_pos_weight` for XGBoost.

### 4. Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- ROC-AUC

Special attention was given to churn recall and F1-score because identifying customers who are likely to leave is important for the business objective.

### 5. Threshold Optimization

Experimented with different classification thresholds instead of relying exclusively on the default threshold of 0.5.

A threshold of 0.6 achieved the highest validation F1-score in the initial XGBoost threshold experiment.

### 6. Feature Importance and Selection

Used XGBoost feature importance to identify the most influential features.

Experimented with different feature subsets:

- 10 features
- 15 features
- 20 features
- 25 features
- 30 features
- 35 features
- 41 features

The 15-feature configuration achieved the best results in the exploratory feature-selection experiment.

### 7. Hyperparameter Tuning

Experimented with different XGBoost configurations by changing:

- `n_estimators`
- `learning_rate`
- `max_depth`
- `subsample`
- `colsample_bytree`

The fifth configuration achieved the highest validation F1-score and recall among the five tested configurations.

Selected configuration:

```python
{
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7
}
```

## Results

### Baseline Model Comparison

| Model | Accuracy | Churn Precision | Churn Recall | Churn F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 76.65% | 55% | 62% | 59% | 0.8175 |
| XGBoost | 75.59% | 53% | 76% | 62% | 0.8395 |

The baseline XGBoost model achieved a higher churn recall and ROC-AUC than Random Forest, despite having slightly lower accuracy.

### Final Tuned Model

Final test metrics: **To be filled in after final evaluation.**

| Metric | Result |
|---|---|
| Accuracy | Pending |
| Precision | Pending |
| Recall | Pending |
| F1-score | Pending |
| ROC-AUC | Pending |

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Jupyter Notebook

## Key Learnings

Through this project, I learned how to:

- Perform exploratory data analysis on a real-world dataset.
- Clean and preprocess numerical and categorical data.
- Handle missing values and class imbalance.
- Train and compare classification models.
- Evaluate models using multiple classification metrics.
- Interpret confusion matrices and ROC curves.
- Understand feature importance and feature selection.
- Optimize classification thresholds.
- Experiment with XGBoost hyperparameters.
- Save a trained model for future predictions.
- Understand the importance of validation data and avoiding data leakage.

## Project Structure

```text
Customer-Churn-Prediction/
│
├── customer_churn.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── customer_churn_model.pkl
├── requirements.txt
└── README.md
```

*Note: The saved model file is generated after running the final training and saving cells.*

## Conclusion

This project provided practical experience with the end-to-end Machine Learning workflow, from understanding and preparing a dataset to training, evaluating, and improving classification models.

It also demonstrated how feature selection, hyperparameter tuning, and threshold optimization can influence model performance.

The project helped strengthen my understanding of classification, model evaluation, and practical Machine Learning experimentation.