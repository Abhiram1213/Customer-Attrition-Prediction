# 📊 Employee Data Analysis & Machine Learning Preprocessing

A comprehensive, production-grade data science project performing end-to-end data auditing, systematic data cleaning, exploratory data analysis (EDA), categorical encoding, feature scaling, and predictive modeling on employee compensation data.

---

## 📁 Repository Structure

```
├── Employee.csv                   # Cleaned & standardized dataset
├── employe.ipynb                  # Fully executed, overhauled primary notebook
├── Employee_Analysis_Report.pdf   # Publication-grade 4-page Executive PDF Report
└── README.md                      # Project documentation & insights
```

---

## 🛠️ Issues in Original Notebook vs. Solutions Implemented

| Area | Original Issue | Modernized Fix |
|---|---|---|
| **Path Portability** | Hardcoded Windows path (`C:/Users/abhir/Downloads/Employee.csv`) | Portable auto-resolving dataset loader |
| **Invalid Values** | `Age = 0` (6 rows) counted as real values, biasing mean | Converted `Age = 0` to `NaN` first, then imputed with median |
| **Data Cleaning Order** | Missing value imputation performed before resolving invalid zeros | Cleaned in strict logical order: Duplicates $\rightarrow$ Invalids $\rightarrow$ Typos $\rightarrow$ Imputation |
| **Typo & Alias Fragmentation** | Duplicate company aliases (`Tata Consultancy Services`, `Infosys Pvt Lmt`, `Congnizant`) and location typos (`Podicherry`) | Standardized aliases to `TCS`, `Infosys`, `CTS` and location to `Pondicherry` |
| **Missing Values** | 24 NaNs left unresolved in `Age` | All features imputed (Median for Age, Mean for Salary, Mode for Company/Place) |
| **Label Encoding** | Destructive in-place loop corrupted continuous `Age` & `Salary` | Applied strictly to categorical columns in dedicated `df_label_encoded` |
| **One-Hot Encoding** | Index alignment mismatch and guessed column names using `unique()` | Implemented `pd.get_dummies` and `sklearn.OneHotEncoder` with `get_feature_names_out()` |
| **Feature Scaling** | Scaled corrupted label-encoded matrix into raw unformatted array | Structured `StandardScaler` ($\mu=0, \sigma=1$) and `MinMaxScaler` ($[0,1]$) DataFrames with comparison plots |
| **Visualizations** | Only 2 basic unstyled plots | 10+ publication-quality Seaborn/Matplotlib visual figures with annotations |
| **Machine Learning** | None | Trained and evaluated Linear Regression, Ridge, Decision Tree, and Random Forest Regressors |
| **Documentation** | 0 Markdown cells, empty cells | 22 rich Markdown cells with mathematical formulas, objectives, and findings |

---

## 📦 Requirements & Environment Setup

To ensure zero warning messages and optimal numerical computation compatibility:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn "numexpr>=2.10.2" reportlab nbformat nbclient
```

---

## 🚀 How to Run Locally

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(or run the pip command above)*
3. Launch Jupyter:
   ```bash
   jupyter notebook employe.ipynb
   ```
