# 📄 Comprehensive Project & Technical Report: Customer Attrition Prediction

**Project Title**: Telecom Customer Attrition Prediction & Retention Analytics  
**Domain**: Telecommunications / Machine Learning / Customer Analytics  
**Author**: Abhiram  
**Repository**: [Customer-Attrition-Prediction](https://github.com/Abhiram1213/Customer-Attrition-Prediction)  

---

## Executive Summary

Customer acquisition costs (CAC) in the telecommunications industry typically outpace customer retention costs by a factor of **5x to 7x**. A regional telecom service provider in California (*Verizonet*) identified a major retention leak: only **43% of newly acquired subscribers** remained active past their first quarter, contributing to an annualized customer churn rate of **26.54%**.

This project establishes an end-to-end Machine Learning pipeline and analytical study to:
1. Diagnose root causes of subscriber dissatisfaction and competitor departures.
2. Formulate a leakage-free preprocessing pipeline.
3. Benchmark 5 supervised classification models using **10-Fold Stratified Cross-Validation**.
4. Deliver the **Champion Gradient Boosting Model** achieving **94.39% Test Accuracy**, **0.890 F1-Score**, and **0.983 ROC-AUC**.
5. Formulate an actionable 4-pillar customer retention playbook.

---

## 1. Problem Statement & Dataset Profile

### 1.1 Dataset Architecture
The dataset (`verizonet_data.csv`) comprises **7,043 subscriber records** across **43 attributes**:
* **Demographics**: Gender, Age, Senior Citizen status, Partner/Dependents status.
* **Services**: Phone service, Multiple lines, Internet connection type (DSL, Fiber Optic, Cable), Online Security, Tech Support, Streaming services.
* **Contractual & Financials**: Contract duration (Month-to-month, 1-Year, 2-Year), Paperless billing, Payment method, Monthly charge, Total regular charges, Total refunds, CLTV.
* **Customer Interaction**: Total service requests logged, Product/Service issues reported, Satisfaction rate (1 to 5).
* **Target Variable**: `Churn Status` (`0` = Retained, `1` = Churned).

### 1.2 Target Class Distribution
* **Retained Customers (0)**: 5,174 (73.46%)
* **Churned Customers (1)**: 1,869 (26.54%)

---

## 2. Exploratory Data Analysis & Diagnostic Insights

| Operational Driver | Key Finding | Churn Impact |
|:---|:---|:---|
| **Support Ticket Velocity** | Customers logging $\ge 2$ support tickets show a dramatic jump in churn. | Churn probability escalates past **75%**. |
| **Contract Duration** | Month-to-Month contracts provide zero barrier to exit. | Account for **88%** of total churn volume. |
| **Internet Architecture** | Fiber Optic and Cable users without Tech Support experience highest churn. | Unresolved technical friction drives competitor departures. |
| **Satisfaction Rating** | Scores of 1 or 2 almost universally convert to account cancellation. | Over **85%** of 1-rated customers churn. |
| **Tenure Vulnerability** | First-year subscribers (0–12 months) represent the peak churn cohort. | Retention stabilizes significantly after month 24. |

---

## 3. Data Preprocessing & Target Leakage Rectification

### 3.1 Eliminating Target Leakage
* In real-world production systems, `Churn Category` and `Churn Reason` are post-event diagnostic notes recorded only **after** a customer cancels.
* Including these fields in training creates direct target leakage and deceptive 100% training accuracy.
* These fields, along with unique ID strings (`Customer ID`, `City`, `Zip Code`), were strictly excluded from the predictive feature matrix $X$.

### 3.2 Scikit-Learn Preprocessing Pipeline
* **Numerical Features (15)**: Imputed with median and scaled using `StandardScaler`.
* **Categorical Features (22)**: Imputed with `'Missing'` placeholder and encoded using `OneHotEncoder(drop='first', handle_unknown='ignore')`.

---

## 4. Machine Learning Benchmarking & Evaluation

### 4.1 Cross-Validation Strategy
All models were benchmarked using **10-Fold Stratified Cross-Validation** on the 80% training split and verified on an independent 20% holdout test set (1,409 customers).

### 4.2 Benchmark Results Table

| Model Architecture | 10-Fold CV Accuracy | 10-Fold CV F1 | 10-Fold CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Decision Tree** (Depth 6) | 88.25% | 0.751 | 0.908 | 87.86% | 0.780 | 0.757 | 0.768 | 0.912 |
| **Random Forest** (200 Trees) | 92.01% | 0.834 | 0.962 | 92.55% | 0.927 | 0.781 | 0.848 | 0.968 |
| **Support Vector Machine** | 92.83% | 0.858 | 0.970 | 93.75% | 0.901 | 0.858 | 0.879 | 0.975 |
| **Logistic Regression** (L2) | 92.92% | 0.862 | 0.972 | 93.83% | 0.902 | 0.861 | 0.881 | 0.976 |
| **Gradient Boosting** (Champion) | **93.43%** | **0.868** | **0.976** | **94.39%** | **0.928** | **0.856** | **0.890** | **0.983** |

---

## 5. Champion Model Diagnostics (Gradient Boosting)

### 5.1 Test Confusion Matrix Breakdown
* **True Negatives ($TN$)**: 1,010 retained customers correctly identified.
* **True Positives ($TP$)**: 320 churned customers successfully identified for intervention.
* **False Positives ($FP$)**: Only 25 retained customers incorrectly flagged (2.4% false alarm rate).
* **False Negatives ($FN$)**: 54 churners undetected.

### 5.2 Top Feature Importance Signals
1. **Total Customer Service Requests**: Strongest predictor of impending departure.
2. **Customer Satisfaction Rate**: Direct measurement of customer sentiment.
3. **Contract Type (Month-to-Month)**: Primary structural vulnerability.
4. **Monthly Charges**: High sensitivity among upper-tier billings.
5. **Product/Service Issues Reported**: Unresolved hardware/connectivity complaints.

---

## 6. Strategic Business Retention Playbook

| Priority | Strategy Pillar | Target Customer Cohort | Concrete Actionable Intervention |
|:---:|:---|:---|:---|
| **P1** | **Support SLA Escalation** | Subscribers opening $\ge 2$ support tickets | Auto-assign dedicated senior technical account managers; provide a $15 bill credit upon resolution. |
| **P2** | **Contract Migration Campaign** | Month-to-Month accounts with tenure $<12$ mos | Offer a 15% discount on an annual contract commitment with guaranteed price lock protection. |
| **P3** | **Network & Hardware Upgrades** | High-charge Fiber Optic & Cable users | Deploy proactive Wi-Fi 6 router upgrades and provide 60 days of complimentary Premium Tech Support. |
| **P4** | **Referral Program Incentives** | Single-line accounts with 0 referrals | Deploy a "Give $25, Get $25" bill credit referral incentive to build sticky network retention. |

---

## 7. Conclusion & Next Steps
By replacing ad-hoc encoding with a leakage-free Scikit-Learn pipeline and selecting Gradient Boosting as the champion model, the business gains an early-warning system capable of catching **86% of potential churners** before contract cancellation.
