# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using Logistic Regression in Python.

## Overview
Fraud is rare compared with legitimate transactions, so the data is heavily imbalanced. This project handles that with under-sampling, then trains a classifier to tell legitimate and fraudulent transactions apart.

## Dataset
- Credit card transactions dataset (`creditcard.csv`): 284,807 transactions, 31 columns
- Features: `Time`, `Amount`, and 28 anonymized features (`V1`-`V28`)
- Target: `Class` (0 = legitimate, 1 = fraud)
- Only 492 transactions (about 0.17%) are fraud, so the data is highly imbalanced
- No missing values

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Jupyter Notebook

## Workflow
1. Load the data and check its structure and missing values
2. Compare legitimate and fraudulent transactions (amount statistics and feature means)
3. Handle the imbalance with under-sampling: 492 random legitimate transactions plus all 492 fraud cases
4. Split into train and test sets (80/20, stratified)
5. Train a Logistic Regression model
6. Evaluate on training and test data
7. Test whether more iterations improve the model

## Results
| Data | Accuracy |
|------|----------|
| Training | 94.2% |
| Test | 92.4% |

Increasing `max_iter` to 1000 raised training accuracy to 95.8%, but test accuracy stayed at 92.4%.

## How to Run
1. Clone the repo:
   git clone git clone https://github.com/Ama-an-nu/Fraud_Credit_card_transaction_model.git
2. Install the dependencies:
   pip install pandas numpy scikit-learn
3. Download `creditcard.csv` (the Kaggle "Credit Card Fraud Detection" dataset) and place it in the project folder.
4. Open `Fraud_Detection.ipynb` and run all cells.

## Future Improvements
- Scale features with `StandardScaler` (it's already imported, and it would fix the convergence warning)
- Report precision, recall and F1 instead of only accuracy
- Try other methods such as SMOTE, Random Forest or XGBoost
