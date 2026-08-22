# 📄 Research & Engineering Project Report: Telecom Customer Churn Prediction
## An In-Depth Econometric & Machine Learning Study on Subscriber Retention Dynamics

**Project Title**: End-to-End Predictive Modeling & Retention Analytics for Telecommunications  
**Author**: Abhiram  
**Repository**: [Customer-Attrition-Prediction](https://github.com/Abhiram1213/Customer-Attrition-Prediction)  
**Target Class**: `Churn Status` (Binary: $0 = \text{Retained}$, $1 = \text{Churned}$)  
**Total Sample Size**: 7,043 Accounts | 43 Operational Features  
**Champion Architecture**: Gradient Boosting Classifier (**Holdout Accuracy: 94.39%**, **F1-Score: 0.890**, **ROC-AUC: 0.983**)  

---

## 1. Executive Summary & Financial Unit Economics

### 1.1 The Macroeconomic Problem
In the telecommunications sector, market saturation and low technical differentiation mean that top-line growth is overwhelmingly dictated by **Net Revenue Retention (NRR)** rather than raw customer acquisition.
* **Customer Acquisition Cost (CAC)**: Industry averages indicate acquiring a new postpaid customer costs between **$300 and $450** (including ad spend, sales commissions, hardware subsidies, and onboarding friction).
* **Customer Retention Cost (CRC)**: Targeted retention interventions (e.g., promotional credits, plan restructuring, VIP technical support) cost between **$30 and $60** per account.
* **Economic Leverage**: Preventing account churn yields an **ROI of over 600%** compared to backfilling churned subscribers through new sales.

### 1.2 Case Study: California Regional Telecom (*Verizonet*)
* Verizonet observed that only **43% of new cohort subscribers** stayed active beyond their first quarter of service.
* The overall customer churn rate escalated to **26.54%** ($1,869$ lost customers out of $7,043$ accounts).
* **Annualized Revenue Impact**: At an average monthly revenue per user (ARPU) of **$65/month**, the annual gross revenue loss from churn exceeds **$1.45 Million** for this cohort alone.

```
                          FINANCIAL IMPACT SUMMARY
┌──────────────────────────────────────┬────────────────────────────────────────┐
│ Metric                               │ Value                                  │
├──────────────────────────────────────┼────────────────────────────────────────┤
│ Total Customer Accounts              │ 7,043                                  │
│ Total Churned Accounts               │ 1,869 (26.54%)                         │
│ Total Retained Accounts              │ 5,174 (73.46%)                         │
│ Average Monthly Charge (Churned)     │ $80.24 / month                         │
│ Average Monthly Charge (Retained)    │ $61.26 / month                         │
│ Estimated Annual Gross Revenue Lost  │ $1,457,750 USD                         │
└──────────────────────────────────────┴────────────────────────────────────────┘
```

---

## 2. Exhaustive Data Dictionary & Feature Engineering

The dataset comprises **43 raw attributes**. Below is the detailed breakdown of every variable:

### 2.1 Customer Identifiers & Demographics
1. **`Customer ID`**: Unique alphanumeric key (e.g., `'8779-QRDMV'`). *Action*: Dropped from predictive feature set to prevent overfitting on unique keys.
2. **`Gender`**: Nominal binary (`'Male'`, `'Female'`). Distribution: $50.4\%$ Male, $49.6\%$ Female. Churn delta across genders is $<0.8\%$, showing no gender bias.
3. **`Age`**: Discrete integer ($19$ to $80$ years, $\mu = 46.5, \sigma = 16.7$). Higher age bands exhibit moderate positive correlation with churn ($r = +0.11$).
4. **`Under 30`**: Nominal binary (`'Yes'`, `'No'`). Younger subscribers exhibit higher mobility and price sensitivity.
5. **`Senior Citizen`**: Nominal binary (`'Yes'`, `'No'`). Seniors account for $16.2\%$ of users but have a **$41.6\%$ churn rate** (vs $23.6\%$ for non-seniors).
6. **`Married`**: Nominal binary (`'Yes'`, `'No'`). Married accounts show lower churn ($19.6\%$) due to shared household accounts.
7. **`Dependents`**: Nominal binary (`'Yes'`, `'No'`). Families with dependents show high retention ($82.5\%$).
8. **`City` / `Zip Code` / `Population`**: Geographic descriptors across 1,106 California localities.

### 2.2 Telephony & Network Subscriptions
9. **`Phone Service`**: Nominal binary (`'Yes'`, `'No'`). $90.3\%$ of customers subscribe to phone services.
10. **`Multiple Lines`**: Nominal binary (`'Yes'`, `'No'`).
11. **`Internet Service`**: Nominal binary (`'Yes'`, `'No'`). $78.3\%$ of users have internet connectivity.
12. **`Internet Type`**: Categorical (`'DSL'`, `'Fiber Optic'`, `'Cable'`, `'No Internet'`). Fiber Optic subscribers exhibit a **$40.7\%$ churn rate**, driven by higher monthly bills and service expectations.
13. **`Avg Monthly GB Download`**: Continuous integer ($0$ to $85$ GB, $\mu = 20.5$).
14. **`Online Security`**: Nominal binary. Unprotected accounts churn at **$41.7\%$** vs **$14.6\%$** for protected accounts.
15. **`Online Backup`**: Nominal binary.
16. **`Device Protection Plan`**: Nominal binary.
17. **`Premium Tech Support`**: Nominal binary. Lack of tech support correlates strongly with high churn ($41.6\%$).
18. **`Streaming TV / Movies / Music`**: Value-added digital entertainment add-ons.
19. **`Unlimited Data`**: Nominal binary.

### 2.3 Contractual, Billing & Financial Metrics
20. **`Contract`**: Categorical (`'Month-to-Month'`, `'One Year'`, `'Two Year'`). Month-to-month users represent **$88.3\%$ of all churners**.
21. **`Paperless Billing`**: Nominal binary. Correlates positively with churn ($33.5\%$ vs $16.3\%$).
22. **`Payment Method`**: Categorical (`'Bank Withdrawal'`, `'Credit Card'`, `'Mailed Check'`). Bank withdrawal users exhibit highest churn ($34.1\%$).
23. **`Monthly Charge`**: Continuous float ($\$18.25$ to $\$118.75$, $\mu = \$64.76$). Median for churners is $\$79.85$ vs $\$64.40$ for retained.
24. **`Total Regular Charges`**: Continuous float ($\$18.80$ to $\$8684.80$). Cumulative billing.
25. **`Total Refunds`**: Continuous float ($\$0.00$ to $\$497.90$). Refunds indicate prior billing or operational friction.
26. **`Total Extra Data Charges`**: Continuous float.
27. **`Total Long Distance Charges`**: Continuous float.
28. **`CLTV (Customer Lifetime Value)`**: Integer estimate ($2,000$ to $6,500$).

### 2.4 Customer Service & Target Variables
29. **`Total Customer Svc Requests`**: Integer ($0$ to $9$). The single most powerful leading indicator of churn.
30. **`Product/Service Issues Reported`**: Integer ($0$ to $6$).
31. **`Customer Satisfaction rate`**: Discrete integer ($1$ to $5$).
32. **`Churn Category` & `Churn Reason`**: Post-churn diagnostic notes (Strictly isolated from predictive modeling to prevent target leakage).
33. **`Churn Status`**: Ground truth target ($0 = \text{Retained}$, $1 = \text{Churned}$).

---

## 3. Data Preprocessing & Target Leakage Prevention

```
                      END-TO-END PREPROCESSING PIPELINE
                               Raw Input Data
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
       [Target & Leakage Drop]                [Feature Segmentation]
  • Churn Category (Post-event)            • 15 Numerical Features
  • Churn Reason (Post-event)              • 22 Categorical Features
  • Customer ID, City, Zip Code                          │
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │ Scikit-Learn ColumnTransformer│
                     ├───────────────────────────────┤
                     │ Numerical Pipeline:           │
                     │  1. SimpleImputer(median)     │
                     │  2. StandardScaler()          │
                     │                               │
                     │ Categorical Pipeline:         │
                     │  1. SimpleImputer('Missing')  │
                     │  2. OneHotEncoder(drop='first'│
                     │        handle_unknown='ignore'│
                     └───────────────┬───────────────┘
                                     ▼
                      Dense Scaled & Encoded Matrix
                                (X_train, X_test)
```

### 3.1 Mathematical Principles of Preprocessing
* **Standard Scaling**:
  $$z_j = \frac{x_j - \mu_j}{\sigma_j}$$
  Ensures that distance-based algorithms (SVM, Logistic Regression) are not dominated by features with large absolute ranges (e.g., Total Regular Charges in thousands vs. Satisfaction Rate in single digits).
* **One-Hot Encoding**: Converts nominal categories into $K-1$ binary columns (using `drop='first'`) to eliminate the dummy variable trap (linear dependency among columns).

---

## 4. Mathematical Foundations of Machine Learning Models

### 4.1 Logistic Regression (L2 Regularized)
Logistic regression models the log-odds of the posterior probability $P(Y=1|X)$ using the sigmoid activation function:
$$P(Y=1|X) = \sigma(w^T X + b) = \frac{1}{1 + e^{-(w^T X + b)}}$$
The objective function minimizes the regularized cross-entropy loss:
$$\min_w \mathcal{L}(w) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right] + \frac{\lambda}{2} \|w\|_2^2$$

### 4.2 Decision Tree Classifier
Decision trees partition the feature space recursively by maximizing Information Gain (or minimizing Gini Impurity):
$$I_G(t) = 1 - \sum_{k=1}^K p(k|t)^2$$
Split selection at node $t$ chooses the feature and threshold that maximizes:
$$\Delta I_G(s, t) = I_G(t) - \left( \frac{N_L}{N} I_G(t_L) + \frac{N_R}{N} I_G(t_R) \right)$$

### 4.3 Random Forest Classifier
Ensemble bagging algorithm combining $B = 200$ de-correlated decision trees:
$$\hat{f}_{\text{rf}}(x) = \frac{1}{B} \sum_{b=1}^B \hat{f}_b(x)$$
* **Variance Reduction Property**: If each individual tree has variance $\sigma^2$ and average correlation $\rho$, the variance of the ensemble is:
  $$\text{Var}(\hat{f}_{\text{rf}}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
  As $B \to \infty$, the second term approaches zero, eliminating model variance while maintaining low bias.

### 4.4 Support Vector Machine (Linear SVM)
Maximizes the geometric margin between positive and negative classes by solving the convex quadratic programming problem:
$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{subject to } y_i (w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

### 4.5 Gradient Boosting Classifier (Champion Architecture)
Gradient boosting constructs an additive model of $M$ decision trees sequentially:
$$F_M(x) = F_0(x) + \sum_{m=1}^M \gamma_m h_m(x)$$
At each iteration $m$, the algorithm computes pseudo-residuals (negative gradient of the loss function):
$$r_{im} = -\left[ \frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$$
A new regression tree $h_m(x)$ is fitted to these residuals, directly optimizing the loss in function space.

---

## 5. Comprehensive Multi-Model Benchmarking

### 5.1 Validation Protocol
* **Training Set**: 80% ($5,634$ records) evaluated with **10-Fold Stratified Cross-Validation**.
* **Test Set**: 20% ($1,409$ holdout records) reserved strictly for final validation.

### 5.2 Performance Comparison Table

| Model Architecture | 10-Fold CV Accuracy | 10-Fold CV F1 | 10-Fold CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Decision Tree** (Max Depth 6) | 88.25% | 0.751 | 0.908 | 87.86% | 0.780 | 0.757 | 0.768 | 0.912 |
| **Random Forest** (200 Trees) | 92.01% | 0.834 | 0.962 | 92.55% | 0.927 | 0.781 | 0.848 | 0.968 |
| **Support Vector Machine** | 92.83% | 0.858 | 0.970 | 93.75% | 0.901 | 0.858 | 0.879 | 0.975 |
| **Logistic Regression** (L2) | 92.92% | 0.862 | 0.972 | 93.83% | 0.902 | 0.861 | 0.881 | 0.976 |
| **Gradient Boosting** (Champion) | **93.43%** | **0.868** | **0.976** | **94.39%** | **0.928** | **0.856** | **0.890** | **0.983** |

---

## 6. Champion Model Diagnostics & Error Decomposition

### 6.1 Holdout Confusion Matrix ($N = 1,409$)

```
                                PREDICTED CLASS
                         Retained (0)      Churned (1)
ACTUAL Retained (0)         1,010              25          (Support: 1,035)
CLASS  Churned (1)             54             320          (Support: 374)
```

* **True Positive Rate (Sensitivity / Recall)**: $\frac{320}{374} = \mathbf{85.56\%}$ (85.6% of departing subscribers detected prior to departure).
* **True Negative Rate (Specificity)**: $\frac{1010}{1035} = \mathbf{97.58\%}$ (Only 2.4% false alarm rate).
* **Positive Predictive Value (Precision)**: $\frac{320}{345} = \mathbf{92.75\%}$ (92.8% of flagged accounts are genuine churn risks).

### 6.2 Top 15 Feature Importances (MDI Analysis)

```
                            RELATIVE FEATURE IMPORTANCES
┌──────────────────────────────────────┬────────────────────────┬─────────────────────┐
│ Feature Name                         │ Relative Importance    │ Cumulative Impact   │
├──────────────────────────────────────┼────────────────────────┼─────────────────────┤
│ Total Customer Service Requests      │ 28.42%                 │ 28.42%              │
│ Customer Satisfaction Rate           │ 16.18%                 │ 44.60%              │
│ Contract (Month-to-Month)            │ 12.54%                 │ 57.14%              │
│ Monthly Charge                       │ 8.71%                  │ 65.85%              │
│ Product/Service Issues Reported      │ 6.14%                  │ 71.99%              │
│ Tenure in Months                     │ 5.32%                  │ 77.31%              │
│ Internet Type (Fiber Optic)          │ 4.25%                  │ 81.56%              │
│ Number of Referrals                  │ 3.88%                  │ 85.44%              │
│ Total Regular Charges                │ 3.12%                  │ 88.56%              │
│ Premium Tech Support (No)            │ 2.76%                  │ 91.32%              │
│ Online Security (No)                 │ 2.45%                  │ 93.77%              │
│ Payment Method (Bank Withdrawal)     │ 2.11%                  │ 95.88%              │
│ Total Extra Data Charges             │ 1.54%                  │ 97.42%              │
│ Senior Citizen (Yes)                 │ 1.35%                  │ 98.77%              │
│ Age                                  │ 1.23%                  │ 100.00%             │
└──────────────────────────────────────┴────────────────────────┴─────────────────────┘
```

---

## 7. Executive Business Retention Playbook & ROI Model

### 7.1 Intervention Economics (Cost-Benefit Analysis)
Assuming an active subscriber base of 10,000 accounts with baseline churn of 26.54% ($2,654$ churners/year):
* **Without ML Model**: 2,654 subscribers churn $\times$ $\$780/\text{year}$ revenue = **$\$2,070,120$ annual loss**.
* **With ML Intervention Model**:
  * Captures $85.6\%$ of churners ($2,272$ at-risk accounts flagged).
  * Intervention Success Rate = $60\%$ retention after VIP discount & ticket resolution ($1,363$ accounts saved).
  * Net Revenue Preserved: $1,363 \times \$780 = \mathbf{\$1,063,140/\text{year}}$.
  * Total Intervention Cost: $2,272 \times \$45 = \$102,240$.
  * **Net Financial Gain**: $\mathbf{\$960,900/\text{year}}$ (**ROI of 940%**).

### 7.2 Actionable 4-Pillar Retention Strategy

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                            STRATEGIC 4-PILLAR RETENTION PLAYBOOK                            │
├────┬─────────────────────────┬───────────────────────────────┬──────────────────────────────┤
│ #  │ Strategy Pillar         │ Target Segment                │ Operational Intervention     │
├────┼─────────────────────────┼───────────────────────────────┼──────────────────────────────┤
│ P1 │ Support SLA Escalation  │ Accounts logging ≥ 2 tickets  │ Automated VIP routing to     │
│    │                         │                               │ senior engineers + $15 credit│
│ P2 │ Contract Migration      │ Month-to-Month (Tenure < 12m) │ 15% rate-lock discount on    │
│    │                         │                               │ annual plan migration        │
│ P3 │ Proactive Hardware QA   │ High-billing Fiber/Cable users│ Complimentary Wi-Fi 6 router │
│    │                         │                               │ + 60 days Tech Support       │
│ P4 │ Viral Referral Program  │ Single-line accounts (0 refs) │ "Give $25, Get $25" bill     │
│    │                         │                               │ credit referral campaign     │
└────┴─────────────────────────┴───────────────────────────────┴──────────────────────────────┘
```

---

## 8. Conclusion
This project establishes a reproducible, production-grade churn prediction engine. By eliminating target leakage, leveraging Scikit-Learn `ColumnTransformer` pipelines, and implementing the **Gradient Boosting Classifier**, the telecommunications provider gains a high-precision early-warning system capable of preserving over **$960K annually in net recurring revenue**.
