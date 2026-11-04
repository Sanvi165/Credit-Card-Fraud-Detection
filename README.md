# Credit Card Fraud Detection

Detects fraudulent credit card transactions using Logistic Regression on a highly imbalanced dataset.

## Dataset
Kaggle "Credit Card Fraud Detection" (ULB): 284,807 transactions, only 492 frauds (0.17%).
Features V1-V28 are PCA-transformed (anonymised), plus Time and Amount.
Download the dataset from Kaggle and place `creditcard_dataset.csv` in the same folder as the notebook.

## Approach
1. Exploratory data analysis (class imbalance, amount distribution, correlations)
2. Stratified 80/20 train-test split on the real data
3. Undersampling applied to the training set only
4. Logistic Regression with StandardScaler (sklearn Pipeline)
5. Evaluation on the untouched, imbalanced test set

## Results (test set: 56,962 transactions, 98 frauds)
- Accuracy: 0.967
- Fraud recall: 0.95
- Fraud precision: 0.05
- ROC-AUC: 0.99

## Key Learning
Accuracy is misleading on imbalanced data. Undersampling before splitting gives unrealistic results,
so I split first and undersampled only the training data. Raising the decision threshold
trades a small drop in recall for far fewer false alarms.

## Tech Stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
