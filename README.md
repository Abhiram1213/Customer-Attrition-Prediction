# 📊 Telecom Customer Attrition Prediction & Retention Analytics

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)]()

> **An end-to-end machine learning system that identifies customer churn risks, isolates operational churn drivers, and prescribes targeted retention interventions for telecommunications providers.**

---

## 📌 1. Executive Summary & Business Context

Customer acquisition in telecommunications typically costs **5× to 7× more** than retaining an existing subscriber. A regional telecommunications provider (*Verizonet*) serving over 1,000 cities across California faced a critical retention bottleneck: despite strong sales volume, newly acquired cohort retention dropped to **43%**, driving an overall customer churn rate of **26.54%**.

This project provides:
1. **Root-Cause Diagnostic**: Identification of operational drivers behind churn (support escalation thresholds, contract structures, and service types).
2. **Leakage-Free ML Pipeline**: Robust feature pipelines using Scikit-Learn `ColumnTransformer` with median/constant imputers, scaling, and categorical encoders.
3. **Benchmarked Predictive Engine**: Systematic comparison of 4 supervised learning models using **5-Fold Stratified Cross-Validation**, with **Gradient Boosting** achieving **94.39% Test Accuracy** and **0.983 ROC-AUC**.
4. **Actionable Retention Playbook**: Concrete business retention strategies mapping customer risk tiers to high-ROI interventions.

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

## 🏗️ 3. Machine Learning Architecture

The pipeline processes raw tabular customer attributes, handles missing values, normalizes numerical features, encodes categorical variables, and scores customer churn risk:

```
   Raw Customer Records (CSV / Real-Time JSON)
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          1. Preprocessing & Leakage Prevention              │
│  • Exclude Post-Churn Diagnostics (Churn Reason, Category)  │
│  • Impute Missing Numerical (Median) & Categorical ('None') │
│  • Standardize Numerics (StandardScaler)                    │
│  • Encode Nominals (OneHotEncoder, drop='first')            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          2. Supervised Classification Engine                │
│  • 5-Fold Stratified Cross-Validation                      │
│  • Candidate Models: Logistic Regression, Decision Tree,   │
│    Random Forest, Gradient Boosting                         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          3. Decision Support & Risk Tiering                 │
│  • Probability ≥ 0.70 ➔ 🔴 High Risk (Immediate Outreach)   │
│  • 0.40 ≤ Prob < 0.70 ➔ 🟡 Moderate Risk (Nurture & Offer)  │
│  • Probability < 0.40 ➔ 🟢 Low Risk (Healthy Account)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 4. Model Benchmarking & Performance

All models were evaluated using **5-Fold Stratified Cross-Validation** on the training split (80%) and tested on an independent holdout set (20% = 1,409 customers).

| Model Architecture | 5-Fold CV Accuracy | 5-Fold CV F1 | 5-Fold CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** (L2) | 92.92% | 0.862 | 0.972 | 93.83% | 0.902 | 0.861 | 0.881 | 0.976 |
| **Decision Tree** (Max Depth 6) | 88.25% | 0.751 | 0.908 | 87.86% | 0.780 | 0.757 | 0.768 | 0.912 |
| **Random Forest** (200 Trees) | 92.01% | 0.834 | 0.962 | 92.55% | 0.927 | 0.781 | 0.848 | 0.968 |
| **Gradient Boosting** (Champion) | **93.43%** | **0.868** | **0.976** | **94.39%** | **0.928** | **0.856** | **0.890** | **0.983** |

### 🏆 Champion Model Performance Breakdown
- **Test Accuracy**: `94.39%` (1,330 correct predictions out of 1,409 test records)
- **Retained (Class 0)**: Precision: `0.95` | Recall: `0.98` | F1: `0.96` (1,010 True Negatives)
- **Churned (Class 1)**: Precision: `0.93` | Recall: `0.86` | F1: `0.89` (320 True Positives, 25 False Positives)
- **ROC-AUC Score**: `0.983`

---

## 🎯 5. Top Predictive Drivers (Feature Importance)

The top factors influencing customer departure ranked by relative feature importance:
1. **Total Customer Service Requests**: Strongest leading indicator of dissatisfaction.
2. **Customer Satisfaction Rate**: Directly proportional to account retention.
3. **Contract Type (Month-to-Month)**: Lowest commitment tier with highest churn propensity.
4. **Monthly Charges**: Price sensitivity threshold across mid-to-high usage tiers.
5. **Product/Service Issues Reported**: Unresolved technical friction.
6. **Internet Connection Type (Fiber Optic / Cable)**: Infrastructure performance sensitivity.
7. **Tenure in Months**: New subscribers (<12 months) represent peak attrition vulnerability.

---

## 💡 6. Strategic Customer Retention Playbook

| Priority | Strategy Pillar | Target Segment | Concrete Actionable Intervention |
|:---:|:---|:---|:---|
| **P1** | **Support Escalation SLA** | Accounts with ≥ 2 tickets | Auto-assign dedicated senior technical account managers; provide a $15 service credit upon 2nd ticket resolution. |
| **P2** | **Contract Migration Campaign** | Month-to-Month users (Tenure < 12m) | Deliver a personalized 15% discount on an annual commitment, locking in recurring revenue. |
| **P3** | **Network & Hardware Upgrades** | High-charge Fiber/Cable users | Offer complimentary Wi-Fi 6 router upgrades and 60 days of free Premium Tech Support. |
| **P4** | **Referral Program Incentives** | Single-line accounts with 0 referrals | Launch "Give $25, Get $25" bill credit referral campaigns to build customer network retention. |

---

## 📁 7. Repository Structure

```
Customer-Attrition-Prediction/
├── Customer Attrition Prediction.ipynb   # Executed & documented Jupyter Notebook
├── demo.py                               # One-command quickstart demonstration script
├── verizonet_data.csv                    # Dataset (7,043 rows × 43 columns)
├── requirements.txt                      # Version-pinned Python dependencies
├── .gitignore                            # Standard Python & OS ignore rules
├── LICENSE                               # MIT License
├── README.md                             # Human-written technical project documentation
├── models/
│   └── churn_prediction_pipeline.joblib  # Serialized production pipeline artifact
└── src/
    ├── __init__.py                       # Package initializer
    ├── data_loader.py                    # Clean data ingestion & leakage prevention
    ├── preprocessor.py                   # Scikit-learn ColumnTransformer pipeline
    ├── train.py                          # CLI training & model benchmarking script
    └── predict.py                        # CLI batch & real-time inference script
```

---

## 🚀 8. Quickstart & Usage Guide

### Prerequisites
- Python 3.9+ installed
- Git

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/Abhiram1213/Customer-Attrition-Prediction.git
cd Customer-Attrition-Prediction

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Option A: Quick Demonstration (Single Command)
Run immediate evaluation and customer risk scoring with targeted retention actions:
```bash
python demo.py
```

### Option B: Interactive Jupyter Notebook
Launch the notebook to inspect exploratory visual plots, cross-validation metrics, and interactive code:
```bash
jupyter notebook "Customer Attrition Prediction.ipynb"
```

### Option C: Retrain Models via CLI
Train the models, benchmark cross-validation scores, and save the champion pipeline artifact:
```bash
python -m src.train --data verizonet_data.csv --output models/churn_prediction_pipeline.joblib
```

### Option D: Run Predictions & Risk Scoring
Generate churn probabilities and risk levels for new customer records:
```bash
python -m src.predict --input verizonet_data.csv --output predictions.csv
```

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).

