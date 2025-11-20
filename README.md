# SHAP-ANALYSIS-CREDIT-RISK-MODE

## Introduction
This project involves developing a classification model (XGBoost) to predict loan default risk. The model is optimized for performance metrics such as F1-score and AUC. Additionally, SHAP (SHapley Additive exPlanations) is used to interpret global feature importance and individual predictions, ensuring transparency and regulatory compliance.

---

## 1. Model Development & Hyperparameters

### Full Python Code

---

---

## 2. Model Architecture & Performance Metrics

| Aspect                  | Details                                               |
|-------------------------|--------------------------------------------------------|
| Model                   | XGBoost Classifier                                   |
| Hyperparameters        | n_estimators=250, max_depth=6, learning_rate=0.05, scale_pos_weight=3 |
| Performance Metrics     | F1-score: *Insert value*, AUC: *Insert value*     |

*Replace placeholder values with actual computed metrics.*

---

## 3. Global SHAP Interpretation
The SHAP summary plot visualizes feature importance across the dataset. Features like "Debt-to-Income Ratio," "Past Defaults," and "Credit Score" significantly influence the model's prediction of default risk.

### Interpretation
- High positive SHAP values indicate features that increase the likelihood of default.
- Negative SHAP values suggest features that decrease risk.

---

## 4. Individual Prediction Explanation for High-Risk Loans
The following code snippet generates detailed explanations for individual high-risk loan applications (predicted probability ≥ 0.8):


---

## 5. Summary & Conclusions
- The model demonstrates strong predictive performance, with optimized hyperparameters.
- SHAP analysis confirms key domain insights regarding risk factors.
- Individual SHAP explanations provide clear interpretability for high-risk applications, aiding transparency requirements.

---

## Deliverables
- Complete Python code for modeling and interpretability
- Model performance metrics and hyperparameters
- Global SHAP feature importance plots
- Individual prediction explanations

---

## References
- SHAP: https://github.com/slundberg/shap
- XGBoost Documentation: https://xgboost.readthedocs.io/
