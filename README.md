Problem statement:  Credit Card Default Prediction

The goal of this project is to predict whether a customer will default on their credit payment in the next month. By using different machine learning classification models, we aim to identify high-risk customers in advance. The models are compared using evaluation metrics like Accuracy, Precision, Recall, F1 Score, ROC-AUC, and MCC to select the best performing model.

Dataset description:

The dataset  contains 30,000 rows and 25 columns,representing historical credit card information of customers. The target variable is default.payment.next.month, where 1 = default and 0 = no default.

Important features include LIMIT_BAL (credit limit), AGE, SEX, EDUCATION, MARRIAGE, repayment history columns like PAY_0 to PAY_6, bill amounts BILL_AMT1 to BILL_AMT6, and payment amounts PAY_AMT1 to PAY_AMT6.

These features are used to predict whether a customer will default in the next month.





| Model               |   Accuracy |   Precision |   Recall |   F1 Score |   ROC-AUC |    MCC |
|---------------------|------------|-------------|----------|------------|-----------|--------|
| Logistic Regression |     0.6813 |      0.3691 |   0.6217 |     0.4632 |    0.7085 | 0.2747 |
| Decision Tree       |     0.7807 |      0.5039 |   0.5418 |     0.5221 |    0.7643 | 0.3805 |
| K-Nearest Neighbors |     0.8092 |      0.6304 |   0.3316 |     0.4346 |    0.7290 | 0.3577 |
| Naive Bayes         |     0.7523 |      0.4513 |   0.5554 |     0.4980 |    0.7251 | 0.3391 |
| Random Forest       |     0.7908 |      0.5252 |   0.5644 |     0.5441 |    0.7758 | 0.4091 |
| XGBoost             |     0.7653 |      0.4761 |   0.6089 |     0.5344 |    0.7765 | 0.3858 |





| Model | Observation |
|-------|------------|
| Logistic Regression | Shows the weakest overall performance (Accuracy = 0.6813, MCC = 0.2747). Although recall is relatively high (0.6217), very low precision (0.3691) results in many false positives and limited discriminative ability (ROC-AUC = 0.7085). |
| Decision Tree | Provides balanced performance (Accuracy = 0.7807, F1 = 0.5221) with moderate precision and recall. Shows reasonable classification strength but less robustness than ensemble models. |
| K-Nearest Neighbors | Achieves highest accuracy (0.8092) but very low recall (0.3316), meaning many positive cases are missed. Strong precision (0.6304) makes it conservative in predicting positives. |
| Naive Bayes | Delivers moderate and stable performance (Accuracy = 0.7523, F1 = 0.4980) with acceptable but not strong predictive power. |
| Random Forest | Most balanced and reliable model (Accuracy = 0.7908, F1 = 0.5441, highest MCC = 0.4091). Provides strong overall predictive stability. |
| XGBoost | Best class discrimination (highest ROC-AUC = 0.7765) with strong recall (0.6089). Effective when identifying positive cases is important. |
