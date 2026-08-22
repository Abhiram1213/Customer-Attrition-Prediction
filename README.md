# 📊 Telecom Customer Attrition Prediction & Retention Analytics

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()


> **A comprehensive, data-driven machine learning analysis diagnosing telecom customer churn patterns, benchmarking supervised learning models with 5-Fold Stratified Cross-Validation, and prescribing actionable customer retention strategies.**

---

## 📌 1. Project Background & Business Context

Customer acquisition in telecommunications typically costs **5× to 7× more** than retaining an existing subscriber. A regional telecommunications company (*Verizonet*) serving over 1,000 cities in California noticed a significant drop in cohort retention: newly acquired subscribers had only a **43%** retention rate by quarter-end, contributing to an overall customer churn rate of **26.54%**.

This project provides an end-to-end analytical study to:
1. **Uncover Churn Root Causes**: Explore operational, service-level, and contractual pain points driving customer departures.
2. **Prevent Data Leakage**: Isolate post-churn diagnostic features (`Churn Category`, `Churn Reason`) from predictive features to ensure true predictive validity.
3. **Benchmark Machine Learning Classifiers**: Evaluate 5 supervised learning architectures using **5-Fold Stratified Cross-Validation**:
   - Logistic Regression (L2 Regularized)
   - Decision Tree Classifier
   - Random Forest Classifier (Ensemble Bagging)
   - Support Vector Machine (Linear SVM)
   - Gradient Boosting Classifier (**Champion Model**)
4. **Formulate a Retention Playbook**: Deliver concrete, targeted business interventions to safeguard recurring subscription revenue.

---

## 🔍 2. Key Exploratory & Business Insights

Analysis of 7,043 customer accounts revealed specific, actionable churn patterns:

```
                          CHURN DRIVER ANALYSIS SUMMARY
┌───────────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Operational Driver            │ Observed Customer Behavior & Churn Impact                   │
├───────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 🚨 Support Ticket Velocity     │ Logging ≥ 2 customer service tickets spikes churn to >70%.  │
│ 📜 Contract Commitment        │ Month-to-Month accounts churn at 4x the rate of 1-2 Yr plans│
│ 🌐 Service Architecture        │ Fiber Optic/Cable without Tech Support leads exit reasons   │
│ ⭐ Customer Satisfaction Score│ Satisfaction scores of 1-2 almost universally exit (>85%)   │
│ 🤝 Referral Network           │ Customers with 0 referrals are 3.5x more likely to churn    │
└───────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 📈 3. Machine Learning Model Benchmarks

All models were evaluated using **5-Fold Stratified Cross-Validation** on the training split (80%) and tested on an independent holdout set (20% = 1,409 customers).

| Model Architecture | 5-Fold CV Accuracy | 5-Fold CV F1 | 5-Fold CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** (L2) | 92.92% | 0.862 | 0.972 | 93.83% | 0.902 | 0.861 | 0.881 | 0.976 |
| **Decision Tree** (Max Depth 6) | 88.25% | 0.751 | 0.908 | 87.86% | 0.780 | 0.757 | 0.768 | 0.912 |
| **Random Forest** (200 Trees) | 92.01% | 0.834 | 0.962 | 92.55% | 0.927 | 0.781 | 0.848 | 0.968 |
| **Support Vector Machine** (SVM) | 92.83% | 0.858 | 0.970 | 93.75% | 0.901 | 0.858 | 0.879 | 0.975 |
| **Gradient Boosting** (Champion) | **93.43%** | **0.868** | **0.976** | **94.39%** | **0.928** | **0.856** | **0.890** | **0.983** |

### 🏆 Champion Model Highlights (Gradient Boosting)
- **Holdout Test Accuracy**: `94.39%` (1,330 correct predictions out of 1,409 test records)
- **Retained (Class 0)**: Precision: `0.95` | Recall: `0.98` | F1: `0.96` (1,010 True Negatives)
- **Churned (Class 1)**: Precision: `0.93` | Recall: `0.86` | F1: `0.89` (320 True Positives, 25 False Positives)
- **ROC-AUC Score**: `0.983`

---

## 🎯 4. Top Churn Drivers (Feature Importance)

1. **Total Customer Service Requests**: The single strongest leading indicator of subscriber frustration.
2. **Customer Satisfaction Rating**: Direct reflection of ongoing relationship health.
3. **Contract Type (Month-to-Month)**: Lowest friction to cancel and highest churn propensity.
4. **Monthly Charges**: High sensitivity among subscribers in top billing tiers.
5. **Product/Service Issues Reported**: Unresolved hardware or connectivity friction.
6. **Internet Connection Type (Fiber Optic / Cable)**: Competitive pressure when tech support is absent.
7. **Tenure in Months**: New subscribers (<12 months) represent peak attrition vulnerability.

---

## 💡 5. Strategic Customer Retention Playbook

| Priority | Strategy Pillar | Target Customer Cohort | Concrete Actionable Intervention |
|:---:|:---|:---|:---|
| **P1** | **Support Escalation SLA** | Accounts with ≥ 2 support tickets | Auto-assign dedicated senior technical account managers; provide a $15 bill credit upon 2nd ticket resolution. |
| **P2** | **Contract Migration Campaign** | Month-to-Month users (Tenure < 12m) | Deliver a personalized 15% discount on an annual commitment, locking in recurring revenue. |
| **P3** | **Network & Hardware Upgrades** | High-charge Fiber/Cable users | Offer complimentary Wi-Fi 6 router upgrades and 60 days of free Premium Tech Support. |
| **P4** | **Referral Program Incentives** | Single-line accounts with 0 referrals | Launch "Give $25, Get $25" bill credit referral campaigns to build customer network retention. |

---

## 📁 6. Repository Layout

```
Customer-Attrition-Prediction/
├── Customer Attrition Prediction.ipynb   # Complete, executed, and documented Jupyter Notebook
├── verizonet_data.csv                    # Dataset (7,043 rows × 43 columns)
├── requirements.txt                      # Environment dependencies
└── README.md                             # Human-written technical project documentation
```

---

## 🚀 7. Quickstart & How to Run

### Prerequisites
- Python 3.9+ installed
- Jupyter Notebook or VS Code with Jupyter extension

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/Abhiram1213/Customer-Attrition-Prediction.git
cd Customer-Attrition-Prediction

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt
```

### Run the Notebook
Open and explore the fully interactive notebook:
```bash
jupyter notebook "Customer Attrition Prediction.ipynb"
```

