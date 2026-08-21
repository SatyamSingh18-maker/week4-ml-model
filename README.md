# Week 4 – Machine Learning Model Development & Evaluation

Data Science Internship Task | YuvaIntern

## Overview
This project builds and evaluates a binary classification model in Python 
to predict whether a breast tumor is malignant or benign, based on 30 
numeric features derived from digitized tissue images.

## Dataset
Breast Cancer Wisconsin (Diagnostic) Dataset — 569 samples, 30 features, 
loaded directly via Scikit-learn (no missing values).

## What Was Done
- **Data Preparation**: Stratified train/test split (75/25) to preserve 
  class balance; feature scaling with `StandardScaler` (fit on training 
  data only, to avoid data leakage).
- **Model**: Logistic Regression — chosen for interpretability and strong 
  baseline performance on binary classification tasks.
- **Evaluation**: Accuracy, precision, recall, F1 score, ROC-AUC, confusion 
  matrix, precision-recall curve, and 5-fold cross-validation.
- **Overfitting Check**: Compared train vs test accuracy (gap of only 0.22 
  percentage points, confirming low overfitting).

## Results
| Metric | Score |
|---|---|
| Accuracy | 98.60% |
| Precision | 98.89% |
| Recall | 98.89% |
| F1 Score | 98.89% |
| ROC-AUC | 0.998 |

## Key Insight
Cell size and shape irregularity features (worst radius, worst concave 
points, worst perimeter) were the strongest predictors of malignancy — 
consistent with established medical understanding. Despite very strong 
metrics, the report explicitly discusses limitations (small dataset, 
class imbalance, single train/test split) before suggesting the model 
would need further validation before any real clinical use.

## Files
| File | Description |
|---|---|
| `Week4_ML_Model_Report.docx` | Full report with methodology, code, and visualizations |
| `week4_analysis.py` | Python script for model training and evaluation |

## Tools Used
Python, Scikit-learn, Pandas, Matplotlib, Seaborn
