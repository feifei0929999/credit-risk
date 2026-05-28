# Credit Risk Prediction Project

## Project Overview

This project uses machine learning models to predict whether a borrower is likely to default on a loan. The goal is to help financial institutions identify high-risk borrowers and improve credit risk management.

## Dataset

**Source:** UCI Machine Learning Repository — German Credit Dataset

* Number of observations: 1000 borrowers
* Number of features: 20

### Target Variable

* `0` = Good credit
* `1` = Bad credit (default risk)

### Features Include

* Credit amount
* Loan duration
* Age
* Employment status
* Credit history
* Housing status

## Data Limitation

This dataset represents borrowers in Germany and may not fully generalize to borrowers in other countries or financial systems.

## Tools Used

* Python
* pandas
* scikit-learn
* XGBoost

## Machine Learning Models

The following machine learning models were implemented and compared:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

## Model Results

### Logistic Regression

* Accuracy: 0.78
* ROC-AUC: 0.80
* Minority Class (1) Recall: 0.80
* Minority Class (1) Precision: 0.56

**Strengths:**
Achieved the highest recall for the minority class (bad credit), which is important in credit risk prediction because missing high-risk borrowers can lead to financial losses. The ROC-AUC score also indicates good overall model performance.

**Weaknesses:**
Lower precision compared to Random Forest, meaning the model produces more false positives. The model also required feature scaling before training.

---

### Random Forest

* Accuracy: 0.77
* ROC-AUC: 0.80
* Minority Class (1) Recall: 0.35
* Minority Class (1) Precision: 0.78

**Strengths:**
Achieved strong overall accuracy and the highest precision for the minority class. The model is also less sensitive to feature scaling.

**Weaknesses:**
Recall for the minority class was significantly lower than Logistic Regression, meaning many high-risk borrowers were not identified correctly.

---

### XGBoost

* Accuracy: 0.715
* ROC-AUC: 0.779
* Minority Class (1) Recall: 0.52
* Minority Class (1) Precision: 0.53

**Strengths:**
Provided a more balanced trade-off between recall and precision for the minority class.

**Weaknesses:**
Lower overall accuracy and ROC-AUC compared to Logistic Regression and Random Forest in this implementation.

## Key Insights

* Logistic Regression with `class_weight='balanced'` performed best for identifying high-risk borrowers.
* Credit amount, age, and loan duration were among the most important predictors of credit risk.
* Recall is especially important in credit risk prediction because false negatives may lead to financial losses.

## Business Interpretation

In real-world lending, failing to identify a high-risk borrower can result in significant financial losses. Therefore, models with higher recall for the default class are often preferred, even if they generate more false positives.

Based on the project results, Logistic Regression was the preferred model because it achieved the strongest recall for identifying borrowers with default risk.

## Future Improvements

Potential future improvements include:

* Hyperparameter tuning for all models
* Resampling techniques such as SMOTE
* Using larger and more diverse datasets
* Feature engineering and model explainability techniques

## Conclusion

This project demonstrates how machine learning can be applied to credit risk prediction using real-world financial data. Although the dataset is limited to Germany, the modeling approach can be extended to broader financial risk analysis problems.

Among the tested models, Logistic Regression provided the best performance for identifying high-risk borrowers due to its strong recall for the minority class.

## Project Structure

```text
credit-risk-prediction/
│
├── data_cleaning.ipynb
├── eda.ipynb
├── model_comparison.ipynb
├── report.pdf
└── README.md
```
