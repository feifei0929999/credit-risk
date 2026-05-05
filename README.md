# Credit Risk Prediction Project

##  Project Overview

This project uses machine learning to predict whether a borrower is likely to default on a loan.
The goal is to help financial institutions identify high-risk borrowers and improve credit risk management.

##  Dataset

* Source: UCI Machine Learning Repository (German Credit Dataset)

* Number of observations: 1000 borrowers

* Number of featrues: 20

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
* XGBoost Classifier

##  Model Results

### Logistic Regression

* Accuracy: 0.78
* ROC-AUC: 0.80
* Minority Class (1) Recall: 0.80
* Minority Class (1) Precision: 0.56
* Strengths: Achieved the highest recall for the minority class (bad credit), which is often crucial in credit risk prediction to minimize false negatives. The ROC-AUC also indicates good overall discriminative power.
* Weaknesses: Lower precision for the minority class compared to Random Forest, meaning it generates more false positives. Required feature scaling for convergence, indicating sensitivity to feature scales.
* best recall (0.80) of all three models

### Random Forest

* Accuracy: 0.77
* ROC-AUC: 0.80
* Minority Class (1) Recall: 0.35
* Minority Class (1) Precision: 0.78
* Strengths: Achieved the highest accuracy among the three and a relatively high precision for the minority class. Less sensitive to feature scaling.
* Weaknesses: Significantly lower recall for the minority class compared to Logistic Regression, indicating it struggles to identify a large portion of actual bad credit cases despite the class_weight='balanced' parameter. This might be due to the inherent nature of tree-based models or require more aggressive tuning.
* best accuracy (0.775) of all three models

### XGBoost

* Accuracy: 0.715
* ROC-AUC: 0.779
* Minority Class (1) Recall: 0.52
* Minority Class (1) Precision: 0.53
* Strengths: Provides a balanced approach with moderate recall and precision for the minority class. Generally robust and powerful for complex datasets.
* Weaknesses: Lower overall accuracy and ROC-AUC compared to the other two models in this specific implementation. Its recall for the minority class is better than Random Forest but significantly lower than Logistic Regression.
* balanced performance of all three models

##  Key Insights

* Logistic Regression with class_weight='balanced' performed the best compared to Random Forest and XGBoost.
* The model has relatively low recall for default cases, meaning some high-risk borrowers are not detected.
* Credit amount, age, and loan duration are the most important predictors of default risk.

##  Feature Importance

Based on the Random Forest model, the most important features for predicting credit risk include:

* Credit amount
* Age
* Loan duration

These variables have the strongest influence on whether a borrower is classified as high-risk or low-risk.

##  Business Interpretation

* Missing high-risk borrowers (false negatives) can lead to financial losses for lenders.
* Improving recall for default cases is important in real-world applications. Depending on the business objective, the trade-off between precision and recall for the minority class would dictate the final model choice. For a conservative approach to credit risk, high recall for the 'bad credit' class is paramount, making Logistic Regression the preferred choice here.

##  Future Improvements

* Improve recall using class weighting, resampling techniques, or exploring further hyperparameter tuning for all models
* Apply the model to larger and more diverse datasets

##  Conclusion

This project demonstrates how machine learning can be applied to credit risk prediction using real-world data, as it can help financial institutions identify high-risk applicants and reduce potential losses.
Although the dataset is limited to Germany, the modeling approach is applicable to broader financial risk problems. When the business objective prioritizes minimizing losses from defaults, the Logistic Regression model with class_weight='balanced' is the recommended choice due to its superior recall for the minority class. Further hyperparameter tuning for all models, especially Random Forest and XGBoost, could be explored to refine their performance and optimize the trade-off between precision and recall based on specific business costs associated with false positives versus false negatives.
