# Customer Churn Prediction

An end-to-end, reproducible Data Science project that predicts which telecom customers are likely to churn,
explains the main churn patterns, and turns the model into a practical retention-targeting tool.

The project follows a standard Data Science workflow across nine notebooks: problem framing → EDA → cleaning →
preprocessing → feature engineering → modelling → evaluation → business insights. Every decision is logged in
[`PROJECT_NOTES.md`](PROJECT_NOTES.md).

---

## Business problem

A subscription business loses recurring revenue whenever a customer cancels. In this dataset, **26.5% of
customers churned**, and they account for **30.5% of total monthly charges**. Retention actions (discounts,
contract offers, support calls) cost money, so the company needs to know **which customers to target first**.

## Objectives

1. Find which customer characteristics are associated with churn.
2. Build a classifier that outputs a churn **probability** for each customer.
3. Evaluate it with metrics that reflect the business cost of missed churners.
4. Turn the results into actionable, honestly qualified recommendations.

## Dataset

- **Source:** IBM *Telco Customer Churn* sample dataset (commonly distributed on Kaggle as `WA_Fn-UseC_-Telco-Customer-Churn.csv`), stored here as `data/data.csv`.
- **Size:** 7,043 customers × 21 columns (one row per customer, snapshot, no dates).
- **Target:** `Churn` — Yes if the customer left within the last month.
- **Features:** demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), account (`tenure`, `Contract`,
  `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`) and services (phone, internet type and six add-ons).
- **Data issues found:** `TotalCharges` stored as text with 11 blanks (all customers with `tenure = 0`), `SeniorCitizen`
  encoded as 0/1. No duplicates, no outliers.

## Tech stack

Python 3.14 · pandas · NumPy · scikit-learn · Matplotlib · seaborn · joblib · JupyterLab

## Project structure

```
├── data/
│   ├── data.csv                  # raw dataset
│   ├── churn_cleaned.csv         # output of notebook 03
│   └── processed/                # stratified train/test split (notebook 05)
├── notebooks/
│   ├── 01_Problem_Understanding.ipynb
│   ├── 02_Basic_EDA.ipynb
│   ├── 03_Basic_Data_Cleaning.ipynb
│   ├── 04_Deep_EDA.ipynb
│   ├── 05_Data_Preprocessing.ipynb
│   ├── 06_Feature_Engineering.ipynb
│   ├── 07_Model_Building.ipynb
│   ├── 08_Model_Evaluation.ipynb
│   └── 09_Business_Insights.ipynb
├── src/
│   └── churn_utils.py            # shared paths, column lists, preprocessing pipeline, plot style
├── models/
│   ├── best_model.pkl            # final pipeline + decision threshold
│   └── tuned_candidates.pkl      # tuned models from notebook 07 (regenerated, git-ignored)
├── PROJECT_NOTES.md              # decision log
├── README.md
└── requirements.txt
```

`src/churn_utils.py` holds only the code that must be identical across notebooks (paths, column lists, the
preprocessing pipeline), so each notebook stays focused on its own phase.

## Workflow and notebook guide

| # | Notebook | What it does |
|---|---|---|
| 01 | Problem Understanding | Business framing, target definition, column dictionary, leakage considerations |
| 02 | Basic EDA | Types, missing/blank values, duplicates, summary stats, target distribution |
| 03 | Basic Data Cleaning | Fixes `TotalCharges` (blank → 0 for tenure-0 customers), recodes `SeniorCitizen`, drops `customerID`, saves clean data |
| 04 | Deep EDA | Univariate, bivariate and multivariate analysis; churn rate by every feature; correlation; business insights |
| 05 | Data Preprocessing | Stratified 80/20 split, scaling + one-hot encoding pipeline fitted on train only, imbalance strategy |
| 06 | Feature Engineering | Tests two engineered features with CV; **rejected**, as they gave no measurable gain |
| 07 | Model Building | Dummy baseline, Logistic Regression, Random Forest, Gradient Boosting; 5-fold CV; GridSearchCV tuning |
| 08 | Model Evaluation | Out-of-fold comparison, model selection, threshold choice, **single** test-set evaluation, model saving |
| 09 | Business Insights | Permutation importance, coefficients, risk-decile targeting, recommendations, limitations |

## Key findings (associations, not proven causes)

- **Contract type** is the strongest signal: month-to-month customers churn at **42.7%** vs **2.8%** on two-year contracts.
- **Early tenure is the riskiest period:** 52.9% churn in the first 6 months vs 6.6% after 5 years.
  New month-to-month customers churn at **51%**.
- **Fiber optic** customers churn at **41.9%** (vs 19.0% DSL) and pay the most.
- Customers **without OnlineSecurity / TechSupport** churn far more, even within the same internet type
  (fiber: 49.4% without TechSupport vs 22.6% with).
- **Electronic check** payers churn at **45.3%** vs 15–19% for other methods.
- `gender`, `PhoneService` and `MultipleLines` show almost no relationship with churn.

## Final model and results

**Logistic Regression** (C = 10) inside a scikit-learn pipeline (StandardScaler + OneHotEncoder), with a
**decision threshold of 0.25** chosen on cross-validated training predictions (highest precision with recall ≥ 0.80).

Model comparison, 5-fold CV ROC-AUC on the training set (tuned):

| Model | CV ROC-AUC |
|---|---|
| Gradient Boosting | 0.8500 ± 0.012 |
| Random Forest | 0.8479 ± 0.010 |
| Logistic Regression | 0.8464 ± 0.013 |
| Dummy baseline | 0.5000 |

The models are within one standard deviation of each other. Following a selection rule fixed before the results,
the simpler and more interpretable Logistic Regression was chosen.

**Final evaluation on the held-out test set (1,409 customers, used once):**

| Metric | Final model (threshold 0.25) | Dummy baseline |
|---|---|---|
| ROC-AUC | **0.842** | 0.500 |
| Recall (churn) | **0.813** — 304 of 374 churners caught | 0.000 |
| Precision (churn) | 0.499 | 0.000 |
| F1 (churn) | 0.619 | 0.000 |
| PR-AUC | 0.629 | 0.265 |
| Accuracy | 0.734 | 0.735 |

The model and the baseline have the *same* accuracy, yet only the model finds churners, which shows why accuracy
is the wrong metric here. Test results closely match cross-validation (recall 0.817, ROC-AUC 0.846), so there is no
sign of overfitting.

**Targeting value:** ranking test customers by predicted risk, the **top 20% contain 49.5% of all churners**,
and the top 10% have an actual churn rate of 73.8% (vs 26.5% on average).

## How to run

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebooks in order (01 → 09), interactively:
jupyter lab
#    or all at once from the command line:
for nb in notebooks/0[1-9]_*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

Notebooks 03, 05, 07 and 08 write the files later notebooks depend on (`churn_cleaned.csv`, the train/test split,
the models), so run them in order. Random seeds are fixed (`random_state=42`); re-running reproduces the same numbers.

**Using the saved model:**

```python
import joblib
import pandas as pd

artifact = joblib.load("models/best_model.pkl")
customers = pd.read_csv("data/processed/test.csv").drop(columns="Churn")   # cleaned format, no target

churn_probability = artifact["pipeline"].predict_proba(customers)[:, 1]
flag_for_retention = churn_probability >= artifact["threshold"]
```

Input must be in the cleaned format of `data/churn_cleaned.csv` (numeric `TotalCharges`, `SeniorCitizen` as Yes/No, no `customerID`).

## Limitations

- **Snapshot data:** no time dimension, so validation uses a random split rather than a time-based one; performance on future customers may differ.
- **Missing drivers:** no data on support interactions, complaints, network quality, price changes or competitor offers.
- **No cost data:** the threshold assumes a missed churner costs more than an unnecessary retention offer.
- **Moderate precision:** about half of the flagged customers would have stayed anyway.
- **Correlated features** (`MonthlyCharges` is 99.9% explained by the service columns; `tenure` ↔ `TotalCharges` ρ = 0.89)
  limit the interpretation of individual coefficients.
- **Correlation, not causation:** the model identifies *who* is at risk, not *why*, nor whether an intervention will work.

## Possible improvements

- Add behavioural and time-based data, and validate on a time-based split.
- Choose the threshold by **expected profit**, using real intervention costs and customer lifetime value.
- Calibrate probabilities if they will be used as exact risk percentages.
- Run A/B tests or uplift modelling to measure which retention actions actually work.
- Monitor drift and retrain periodically after deployment.
