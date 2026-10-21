# Project Notes — Customer Churn Prediction

Master log of what was done, why, and what was found. All numbers come from the executed notebooks.

**Dataset:** IBM Telco Customer Churn (`data/data.csv`) — 7,043 customers × 21 columns, one row per customer.
**Target:** `Churn` (Yes = left within the last month). **Churn rate:** 26.5%.

---

## Phase 1 — Problem Understanding (`01`)

- **What:** Framed the task as binary classification; documented every column; checked values against the documentation.
- **Why:** Retention is cheaper than acquisition; the business needs a *ranked list* of at-risk customers.
- **Decision:** Model outputs a **probability**, not just Yes/No.
  **Reason:** A limited retention budget means customers must be prioritised; the threshold can follow the budget.
- **Observation:** Churners are 26.5% of customers but **30.5% of monthly revenue** → they tend to be on pricier plans.
- **Leakage check:** No column is recorded after churn. Snapshot data, no dates → random (stratified) split.

## Phase 2 — Basic EDA (`02`)

- **What:** Shape, dtypes, nulls, blank strings, duplicates, unique values, summary statistics, target distribution.
- **Problems found:**
  - `TotalCharges` stored as **text** with **11 blank strings** (hidden missing values; `isna()` shows 0).
  - `SeniorCitizen` encoded 0/1 while other binary columns use Yes/No.
  - `customerID` is an identifier (7,043 unique values).
- **Observation:** No duplicate rows/IDs. 22 rows share a profile with another customer (different IDs → kept).
- **Observation:** Always predicting "No" gives 73.5% accuracy → accuracy is not a useful metric.

## Phase 3 — Basic Data Cleaning (`03`)

- **Decision:** Blank `TotalCharges` → **0** (not dropped, not mean/median-imputed).
  **Reason:** All 11 blanks have `tenure = 0` (new, never-billed customers), and elsewhere
  `TotalCharges ≈ tenure × MonthlyCharges` (corr 0.9996). Median imputation would invent a ~1,400 bill for brand-new customers.
- **Decision:** `SeniorCitizen` 0/1 → No/Yes. **Reason:** It is a category; consistent encoding with other flags.
- **Decision:** Keep "No internet service" / "No phone service" levels.
  **Reason:** Verified to match `InternetService` / `PhoneService` exactly → consistent, not errors.
- **Decision:** Drop `customerID`. **Reason:** No predictive meaning; would only let a model memorise rows.
- **Checked, nothing removed:** no negative values, no whitespace/case inconsistencies, **no IQR outliers**.
- **Output:** `data/churn_cleaned.csv` (7,043 × 20, no missing values).

## Phase 4 — Deep EDA (`04`)

- **Key findings (associations, not causes):**
  - Contract: month-to-month **42.7%** churn vs one-year 11.3% vs two-year **2.8%** (strongest signal).
  - Tenure: **52.9%** churn in months 0–6 → **6.6%** at 61–72 months; non-linear decline.
  - Fiber optic **41.9%** vs DSL 19.0% vs no internet 7.4%; fiber is also the most expensive (median 91.7).
  - No OnlineSecurity / TechSupport ≈ 42% churn vs ~15% with them; holds *within* fiber (49.4% vs 22.6%).
  - Electronic check **45.3%** vs 15–19% for other payment methods.
  - Seniors, no partner/dependents, paperless billing → higher churn.
  - `gender`, `PhoneService`, `MultipleLines` → almost no difference (< 4 points).
- **Multivariate:** Contract and tenure both matter: new month-to-month customers churn at **51%** (n = 1,994),
  while two-year contracts stay at 0–3% in every tenure group.
- **Correlation:** `tenure` ↔ `TotalCharges` Spearman **0.89** → multicollinearity (relevant for interpreting LR).
- **Implications:** stratified split; churn-focused metrics; candidate features `tenure_group`, add-on count.

## Phase 5 — Data Preprocessing (`05`)

- **Target:** Yes → 1, No → 0.
- **Split:** 80/20, `stratify=y`, `random_state=42` → train 5,634 / test 1,409, both 26.54% churn.
  Saved to `data/processed/` so every later notebook uses the identical split.
- **Validation strategy:** 5-fold stratified CV on train (no separate validation set). Test used once, at the end.
- **Pipeline:** `StandardScaler` (3 numeric) + `OneHotEncoder(drop="if_binary", handle_unknown="ignore")` (16 categorical) → 40 features.
  - One-hot because categories are nominal. Scaling needed for regularised logistic regression.
  - No imputer: no missing values remain after cleaning.
- **Leakage prevention:** Preprocessor is fitted on training data only. Inside a `Pipeline` it is re-fitted on each
  CV training fold. Verified: scaler means = train means, not full-data means.
- **Decision (imbalance):** No SMOTE / resampling. Try `class_weight` as a hyperparameter + tune the threshold.
  **Reason:** Imbalance is moderate (~1,500 churners in train); avoids synthetic data and extra dependencies.

## Phase 6 — Feature Engineering (`06`)

- **Tested:** `num_addon_services` (count of 6 add-ons), `tenure_group` (0–12 / 13–24 / 25–48 / 49+ months).
  Rejected without testing: `TotalCharges / tenure` (≈ `MonthlyCharges`, redundant).
- **Result (5-fold CV ROC-AUC):** LR 0.8462 → 0.8465; RF 0.8446 → 0.8451. Gains (< 0.001) are far below the fold std (~0.01).
- **Decision:** **Do not use** the engineered features. **Reason:** No measurable gain; the original one-hot columns plus
  tenure/contract already carry the information. `num_addon_services` is non-monotonic (the 0 group includes no-internet customers).

## Phase 7 — Model Building (`07`)

- **Models:** Dummy (most frequent) baseline, Logistic Regression, Random Forest, Gradient Boosting — all in the same pipeline, same folds.
- **Primary metric:** ROC-AUC (threshold-independent ranking). PR-AUC, precision, recall, F1 also reported.
- **Selection rule (set before results):** best CV ROC-AUC, but prefer the simpler model if the gap < fold std.
- **Baseline:** Dummy → accuracy 0.735, recall 0, ROC-AUC 0.5.
- **Tuning (GridSearchCV, ROC-AUC):**

  | Model | Best params | CV ROC-AUC |
  |---|---|---|
  | Gradient Boosting | lr 0.03, depth 2, 300 trees | 0.8500 ± 0.012 |
  | Random Forest | depth 6, min_leaf 10, max_features 0.5 | 0.8479 ± 0.010 (default was 0.823 → overfitting fixed) |
  | Logistic Regression | C = 10, class_weight None | 0.8464 ± 0.013 |

- **Observation:** `class_weight="balanced"` was never selected — it mostly shifts probabilities, which ROC-AUC ignores.
- **Problem handled:** Nested parallelism warnings (RF `n_jobs=-1` inside `GridSearchCV(n_jobs=-1)`) → parallelise only the outer loop.
  Dummy precision "ill-defined" warning → `zero_division=0` (explicitly 0).

## Phase 8 — Model Evaluation (`08`)

- **Method:** Out-of-fold predictions (`cross_val_predict`, same folds) to compare models and choose the threshold — no test data used.
- **Comparison (OOF, threshold 0.5):** ROC-AUC 0.846–0.849, curves overlap; all models catch only ~50–55% of churners at 0.5.
- **Decision:** Final model = **Logistic Regression**.
  **Reason:** Within noise of Gradient Boosting; highest recall/F1 at 0.5; interpretable; simple to maintain.
- **Business framing:** False negative = lost customer (assumed costlier); false positive = wasted offer. No cost data → assumption.
- **Decision:** Threshold **0.25** = highest precision with OOF recall ≥ 0.80.
  **Reason / honest note:** I first used *max F2*, which picked 0.15. That flags **55%** of all customers (OOF), which is
  too broad for a targeted campaign. The rule was changed based on that OOF table. The test set had already been
  scored once at 0.15 by then, but the test numbers did not drive the change.
- **Final test results (threshold 0.25):**

  | Metric | Value |
  |---|---|
  | ROC-AUC | 0.842 |
  | PR-AUC | 0.629 |
  | Recall | 0.813 (304 of 374 churners caught) |
  | Precision | 0.499 (305 false positives) |
  | F1 | 0.619 |
  | Accuracy | 0.734 (Dummy: 0.735!) |

- **Overfitting check:** Test ≈ CV (recall 0.813 vs 0.817; ROC-AUC 0.842 vs 0.846).
- **Saved:** `models/best_model.pkl` = `{pipeline, threshold, model_name, feature_columns}`. Trained on the train split only,
  so it matches the reported results. Loads with scikit-learn alone (verified from outside the project).

## Phase 9 — Business Insights (`09`)

- **Permutation importance (test ROC-AUC drop):** tenure 0.181, MonthlyCharges 0.118, InternetService 0.117, Contract 0.035;
  gender / PhoneService / Partner ≈ 0.
- **Coefficients:** Fiber optic raises odds ×3.2; two-year contract, DSL, tenure lower risk.
- **Problem found:** `MonthlyCharges` coefficient is *negative* despite churners paying more.
  **Cause (verified):** service columns explain **R² = 0.9988** of `MonthlyCharges` → multicollinearity.
  **Handling:** Don't interpret individual numeric coefficients; base insights on EDA + permutation importance.
- **Risk deciles (test):** top 10% → 73.8% actual churn; top 20% hold **49.5%** of churners; top 30% hold 65.0%;
  bottom 3 deciles 1–3%.
- **Recommendations (hypotheses to A/B test):** onboarding + contract-upgrade incentives for new month-to-month customers;
  investigate the fiber experience/pricing; promote security/tech support; encourage automatic payment; target by risk rank.

## Interview quick-recall

- Why not accuracy? Dummy = 73.5% accuracy with 0 churners caught; the final model has the *same* accuracy but catches 81%.
- Leakage prevention: stratified split first → all fitting inside a Pipeline → CV on train only → test used once.
- Why LR over GB? Gap of 0.0036 AUC < 1 std; interpretability; rule decided before the results.
- Why threshold 0.25? Business wants recall ≥ 80%; chosen on OOF predictions, not on test.
- Biggest limitation: snapshot data with no behavioural/time/cost information → associations only, random split.
