# 📊 Comprehensive Technical & Business Analysis Report
## Telecom Customer Attrition Prediction & Strategic Retention Engine

**Project Title**: End-to-End Machine Learning Pipeline for Telecom Customer Churn Forecasting  
**Author**: Abhiram  
**Repository**: [Customer-Attrition-Prediction](https://github.com/Abhiram1213/Customer-Attrition-Prediction)  
**Dataset Size**: 7,043 Customer Records | 43 Raw Attributes  
**Champion Model**: Gradient Boosting Classifier (Holdout Accuracy: **94.39%**, ROC-AUC: **0.983**)  

---

## 1. Executive Summary & Problem Formulation

### 1.1 Business Context
In subscription-based industries such as telecommunications, Customer Lifetime Value (CLTV) and Net Revenue Retention (NRR) serve as the primary drivers of sustainable valuation. Empirical industry benchmarks demonstrate that acquiring a new telecom subscriber is **5 to 7 times more expensive** than retaining an existing subscriber due to upfront customer acquisition costs (CAC)—including marketing expenditures, sales commissions, SIM provisioning, and equipment subsidies.

A California-based telecommunications service provider (*Verizonet*) operating across 1,000+ cities and zip codes observed a structural retention vulnerability:
* **Quarterly Retention Collapse**: Only **43% of newly onboarded customers** remained subscribed at the conclusion of their first quarter.
* **Annualized Churn Rate**: The baseline churn rate across all cohorts escalated to **26.54%** ($1,869$ departed subscribers out of $7,043$ total accounts).
* **Financial Drag**: The resulting subscriber contraction elevated churn-replacement marketing expenses, eroded average revenue per user (ARPU), and threatened long-term operating margins.

### 1.2 Core Project Objectives
1. **Root-Cause Diagnostic Analysis**: Perform exploratory data analysis to isolate operational, contractual, and technical drivers of customer dissatisfaction.
2. **Target Leakage Prevention**: Establish rigorous data pipeline boundaries, isolating post-churn diagnostic features (`Churn Category`, `Churn Reason`) from predictive training.
3. **Multi-Model Machine Learning Benchmarking**: Train, tune, and cross-validate candidate algorithms across **10 Stratified Folds** to identify the most generalizable predictive engine.
4. **Actionable Retention Playbook**: Map machine learning risk probabilities into high-ROI business interventions for executive and operational teams.

---

## 2. Dataset Architecture & Feature Categorization

The dataset comprises **7,043 records** and **43 attributes**, structured into 5 functional modalities:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                TELECOM DATASET FEATURE MODALITIES                           │
├──────────────────────────┬──────────────────────────────────────────────────────────────────┤
│ Modality                 │ Attributes                                                       │
├──────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ 1. Customer Identifiers  │ Customer ID, City, Zip Code, Population                          │
│ 2. Demographics          │ Gender, Age, Under 30, Senior Citizen, Married, Dependents       │
│ 3. Service Subscriptions │ Phone Service, Multiple Lines, Internet Service, Internet Type,  │
│                          │ Online Security, Online Backup, Device Protection Plan,          │
│                          │ Premium Tech Support, Streaming TV/Movies/Music, Unlimited Data  │
│ 4. Contract & Financials │ Contract, Paperless Billing, Payment Method, Monthly Charge,     │
│                          │ Total Regular Charges, Total Refunds, Total Extra Data Charges,  │
│                          │ Total Long Distance Charges, CLTV (Customer Lifetime Value)      │
│ 5. Operational Sentiment │ Total Customer Svc Requests, Product/Service Issues Reported,    │
│                          │ Customer Satisfaction rate (1-5), Churn Category, Churn Reason,  │
│                          │ Churn Status (Binary Target: 0 = Retained, 1 = Churned)          │
└──────────────────────────┴──────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Cleaning & Pipeline Engineering

### 3.1 Resolving Target Leakage (Crucial Methodological Step)
* **The Vulnerability**: In raw operational datasets, columns such as `Churn Category` (*"Competitor"*, *"Price"*, *"Dissatisfaction"*) and `Churn Reason` (*"Competitor offered higher speeds"*) are logged by customer service agents **only when an account is being terminated**.
* **The Failure Mode**: Including these attributes during model training introduces **direct target leakage**. A classifier easily learns the trivial rule $\text{Churn Reason} \neq \text{'None'} \implies \text{Churn} = 1$, achieving an artificial 100% training accuracy that completely collapses when scoring active, non-churned customers.
* **The Solution**: Strict removal of `Churn Category`, `Churn Reason`, and high-cardinality unique keys (`Customer ID`, `City`, `Zip Code`) from the feature matrix $X$.

### 3.2 Imputation & Feature Transformation Pipeline
We constructed a Scikit-Learn `ColumnTransformer` pipeline:
* **Numerical Pipeline (15 features)**:
  * Imputation: `SimpleImputer(strategy='median')` to handle missing entries without outlier distortion.
  * Normalization: `StandardScaler()` standardizing features to zero mean and unit variance:
    $$z = \frac{x - \mu}{\sigma}$$
* **Categorical Pipeline (22 features)**:
  * Imputation: `SimpleImputer(strategy='constant', fill_value='Missing')`.
  * Encoding: `OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)` eliminating multicollinearity (dummy variable trap) while gracefully handling unseen categorical levels during inference.

---

## 4. Exploratory Data Analysis & Empirical Insights

### 4.1 Target Class Distribution
* **Retained Customers ($y = 0$)**: $5,174$ ($73.46\%$)
* **Churned Customers ($y = 1$)**: $1,869$ ($26.54\%$)
* **Imbalance Ratio**: $\approx 2.77 : 1$. Stratified sampling was mandated across all cross-validation folds to preserve class proportions.

### 4.2 Support Ticket Velocity (The Primary Tipping Point)
* Subscribers logging **0 or 1 customer service request** maintain a churn rate under **$14.2\%$**.
* Upon logging the **2nd support request**, churn likelihood surges past **$75.4\%$**.
* Accounts with $\ge 3$ tickets exhibit an **$88.9\%$** churn probability, indicating that repeat technical complaints reflect unresolved service friction.

### 4.3 Contractual Lock-in & Switching Costs
* **Month-to-Month Contracts**: Characterized by zero switching costs, this group represents **$88.3\%$ of total churn volume**.
* **1-Year & 2-Year Contracts**: Churn rates drop below **$7.8\%$** and **$2.3\%$** respectively, proving that long-term contracts serve as the strongest structural defense against churn.

### 4.4 Internet Connection Type & Value-Added Bundles
* **Fiber Optic & Cable** users experience elevated churn when unbundled from **Online Security** and **Premium Tech Support**.
* Subscribers with active tech protection packages churn at less than one-third the rate of unprotected subscribers ($11.4\%$ vs $38.7\%$).

### 4.5 Tenure Cohort Dynamics
Segmenting customer tenure into actionable business intervals reveals the highest risk window:
* **0–12 Months (New Subscribers)**: Highest concentration of departures ($>52\%$ of all churners).
* **12–24 Months (Early Stage)**: Moderate attrition risk ($22\%$).
* **24–48 Months (Established)**: Low attrition risk ($16\%$).
* **Over 48 Months (Loyal Cohort)**: Ultra-low attrition risk ($<10\%$).

---

## 5. Machine Learning Benchmarking & Evaluation

### 5.1 Validation Strategy
Models were evaluated using **10-Fold Stratified Cross-Validation** on the training split (80% = 5,634 samples) and validated on an independent holdout test set (20% = 1,409 samples).

### 5.2 Mathematical Formulation of Evaluation Metrics
* **Accuracy**: $\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$
* **Precision**: $\text{Prec} = \frac{TP}{TP + FP}$ (Fraction of predicted churners who were true churners)
* **Recall (Sensitivity)**: $\text{Rec} = \frac{TP}{TP + FN}$ (Fraction of actual churners successfully identified)
* **F1-Score**: $\text{F1} = 2 \times \frac{\text{Prec} \times \text{Rec}}{\text{Prec} + \text{Rec}}$ (Harmonic mean balancing precision and recall)
* **ROC-AUC**: Area under the True Positive Rate vs. False Positive Rate curve across all classification thresholds.

### 5.3 Benchmark Comparison Table (1,409 Holdout Test Records)

| Model Architecture | 10-Fold CV Accuracy | 10-Fold CV F1 | 10-Fold CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Decision Tree Classifier** | 88.25% | 0.751 | 0.908 | 87.86% | 0.780 | 0.757 | 0.768 | 0.912 |
| **Random Forest Classifier** (200 Trees) | 92.01% | 0.834 | 0.962 | 92.55% | 0.927 | 0.781 | 0.848 | 0.968 |
| **Support Vector Machine** (Linear SVM) | 92.83% | 0.858 | 0.970 | 93.75% | 0.901 | 0.858 | 0.879 | 0.975 |
| **Logistic Regression** (L2 Regularized) | 92.92% | 0.862 | 0.972 | 93.83% | 0.902 | 0.861 | 0.881 | 0.976 |
| **Gradient Boosting Classifier** (Champion) | **93.43%** | **0.868** | **0.976** | **94.39%** | **0.928** | **0.856** | **0.890** | **0.983** |

---

## 6. Champion Model Diagnostics (Gradient Boosting)

### 6.1 Holdout Confusion Matrix Breakdown

```
                             PREDICTED CLASS
                      Retained (0)     Churned (1)
ACTUAL Retained (0)      1,010              25        (Total: 1,035)
CLASS  Churned (1)          54             320        (Total: 374)
```

* **True Negatives ($TN = 1,010$)**: $97.6\%$ of retained accounts correctly classified.
* **True Positives ($TP = 320$)**: $85.6\%$ of all actual churners successfully intercepted before account cancellation.
* **False Positives ($FP = 25$)**: Extremely low false alarm rate ($2.4\%$), preventing wasteful customer discount distributions.
* **False Negatives ($FN = 54$)**: Only $3.8\%$ of test records were missed churners.

### 6.2 Top Feature Importances (MDI Analysis)
1. **Total Customer Service Requests ($28.4\%$)**: Leading indicator of account distress.
2. **Customer Satisfaction Rate ($16.2\%$)**: Direct metric of subscriber sentiment.
3. **Contract Type (Month-to-Month) ($12.5\%$)**: Primary contractual vulnerability.
4. **Monthly Charge ($8.7\%$)**: Price sensitivity threshold.
5. **Product/Service Issues Reported ($6.1\%$)**: Unresolved hardware/line friction.
6. **Tenure in Months ($5.3\%$)**: Account maturity indicator.

---

## 7. Strategic 4-Pillar Customer Retention Playbook

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                            EXECUTIVE CUSTOMER RETENTION PLAYBOOK                            │
├────┬─────────────────────────┬───────────────────────────────┬──────────────────────────────┤
│ #  │ Strategy Pillar         │ Target Segment                │ Operational Intervention     │
├────┼─────────────────────────┼───────────────────────────────┼──────────────────────────────┤
│ P1 │ Support SLA Escalation  │ Accounts logging ≥ 2 tickets  │ Automated VIP routing to     │
│    │                         │                               │ senior engineers + $15 credit│
│ P2 │ Contract Rate-Lock      │ Month-to-Month (Tenure < 12m) │ 15% rate-lock discount on    │
│    │                         │                               │ annual plan migration        │
│ P3 │ Proactive Hardware QA   │ High-billing Fiber/Cable users│ Complimentary Wi-Fi 6 router │
│    │                         │                               │ + 60 days Tech Support       │
│ P4 │ Viral Referral Program  │ Single-line accounts (0 refs) │ "Give $25, Get $25" bill     │
│    │                         │                               │ credit referral campaign     │
└────┴─────────────────────────┴───────────────────────────────┴──────────────────────────────┘
```

---

## 8. Conclusion
By eliminating target leakage, deploying a production-ready preprocessing pipeline, and leveraging the **Gradient Boosting Model** (Test Accuracy: **94.39%**, ROC-AUC: **0.983**), telecom leadership can proactively safeguard over **85% of at-risk subscription revenue**.
