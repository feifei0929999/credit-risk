# Credit Risk Prediction Project

##  Project Overview

This project uses machine learning to predict whether a borrower is likely to default on a loan.  
The goal is to help financial institutions identify high-risk borrowers and make better lending decisions.

##  Dataset

* Source: LendingClub / Kaggle Credit Risk Dataset
* Target variable: loan default status
* Features include:
  * income
  * loan amount
  * interest rate
  * loan grade
  * employment length
  * debt-to-income ratio

##  Tools Used

* Python
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

##  Exploratory Data Analysis

The project explores:

* Distribution of loan amounts
* Default vs non-default borrowers
* Relationship between interest rate and default risk
* Correlation between numerical variables
* Default rate across different borrower groups

##  Machine Learning Models

The following models are used:

* Logistic Regression
* Random Forest Classifier

##  Model Evaluation

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC score
* Confusion matrix

##  Key Insights

* Borrowers with higher interest rates tend to have higher default risk.
* Lower income and higher debt-to-income ratio may increase the probability of default.
* Credit risk prediction is important because false negatives can cause financial losses for lenders.

##  Future Improvements

* Add XGBoost model
* Improve feature engineering
* Tune hyperparameters
* Handle class imbalance using SMOTE or class weights
* Build an interactive dashboard

##  Business Value

This project demonstrates how data science can support credit risk management by identifying risky borrowers before loans are approved.
