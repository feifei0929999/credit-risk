# Credit Risk Prediction Project

##  Project Overview

This project uses machine learning to predict whether a borrower is likely to default on a loan.
The goal is to help financial institutions identify high-risk borrowers and improve credit risk management.

##  Dataset

* Source: UCI Machine Learning Repository (German Credit Dataset)

* Number of observations: 1000 borrowers

* Target variable:

  * 0 = good credit
  * 1 = bad credit (default)

* Features include:

  * credit amount
  * loan duration
  * age
  * employment status
  * credit history
  * housing status

##  Data Limitation

This dataset represents borrowers in Germany and may not generalize to other countries or financial systems.

##  Tools Used

* Python
* pandas
* scikit-learn

##  Machine Learning Models

The following models were implemented:

* Logistic Regression
* Random Forest Classifier

##  Model Results

### Logistic Regression

* Accuracy: 0.78
* ROC-AUC: 0.80

### Random Forest

* Accuracy: 0.77
* ROC-AUC: 0.80

##  Key Insights

* Logistic Regression performed slightly better than Random Forest.
* The model has relatively low recall for default cases, meaning some high-risk borrowers are not detected.
* Credit amount, age, and loan duration are the most important predictors of default risk.

##  Business Interpretation

* Missing high-risk borrowers (false negatives) can lead to financial losses for lenders.
* Improving recall for default cases is important in real-world applications.

##  Future Improvements

* Improve recall using class weighting or resampling techniques
* Try advanced models such as XGBoost
* Apply the model to larger and more diverse datasets

##  Conclusion

This project demonstrates how machine learning can be applied to credit risk prediction using real-world data.
Although the dataset is limited to Germany, the modeling approach is applicable to broader financial risk problems.
