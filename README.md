# Credit Scoring Model

A machine learning pipeline that classifies customers into **Low / Medium / High Risk**
categories based on financial history and transaction behavior, following the
project brief (Task 1: Credit Scoring Model).

## Files

| File | Description |
|---|---|
| `credit_scoring_model.py` | End-to-end pipeline: data generation, preprocessing, feature engineering, training, evaluation |
| `credit_data.csv` | The dataset used (6,000 customers, 13 raw columns) |
| `model_comparison.csv` / `.png` | Accuracy, Precision, Recall, F1, ROC-AUC for all 4 models |
| `confusion_matrix.png` | Confusion matrix for the best-performing model |
| `class_distribution.png` | Distribution of Low/Medium/High risk customers |
| `best_model_Logistic_Regression.pkl` | Saved trained model (highest F1-score) |
| `scaler.pkl`, `label_encoder.pkl`, `feature_columns.json` | Preprocessing artifacts needed to score new customers |

## About the data

No dataset was provided, so the script generates a **realistic synthetic dataset**
(6,000 customers) whose features and correlations mirror real bank/credit-bureau
data: income, existing debts, loan history, payment behavior, credit utilization,
and account balance, plus a few supporting fields (age, employment length, credit
history length, open credit lines). The risk label is derived from a latent risk
score (built from debt-to-income, utilization, past defaults, payment punctuality,
etc.) plus random noise, so classes overlap the way real-world risk tiers do —
this is why accuracy lands around 70-75% rather than a suspicious ~99%.

**To use your own data:** replace the `generate_credit_dataset()` call with
`pd.read_csv("your_file.csv")`, keeping the same column names (or update
`FEATURE_COLUMNS`), and a target column called `risk_category` with values
`"Low Risk"`, `"Medium Risk"`, `"High Risk"`.

## Pipeline steps

1. **Preprocessing** — duplicate/missing-value checks, label encoding of the
   target, train/test split (80/20, stratified), feature scaling for Logistic
   Regression.
2. **Feature engineering** — derived ratios that are strong credit-risk signals:
   - `debt_to_income_ratio`
   - `default_rate` (defaults ÷ previous loans)
   - `balance_to_income_ratio`
   - `credit_line_utilization_score` (utilization weighted by late-payment rate)
   - `is_new_credit_history` (thin-file flag)
3. **Models trained** — Logistic Regression, Decision Tree, Random Forest, XGBoost.
4. **Evaluation** — Accuracy, macro Precision, macro Recall, macro F1-Score, and
   macro ROC-AUC (one-vs-rest, since this is a 3-class problem).

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.744 | 0.753 | 0.742 | **0.747** | **0.894** |
| Random Forest | 0.729 | 0.744 | 0.725 | 0.733 | 0.881 |
| XGBoost | 0.724 | 0.731 | 0.722 | 0.726 | 0.880 |
| Decision Tree | 0.679 | 0.695 | 0.675 | 0.683 | 0.852 |

**Best model: Logistic Regression** (highest F1-score and ROC-AUC). Its strong
performance suggests the risk signal is largely linear/additive across the
engineered ratio features — which also makes it the most interpretable choice
for a real credit-decisioning use case (regulators generally prefer explainable
models like logistic regression for credit decisions).

All models struggle most with the **Medium Risk** class, which is expected:
it's the transition zone between clearly-good and clearly-bad borrowers and
naturally overlaps with its neighbors.

## How to run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
python credit_scoring_model.py
```

## Ideas for extending this project

- Swap in a real dataset (e.g., the UCI German Credit or Lending Club datasets)
- Hyperparameter tuning via `GridSearchCV` / `Optuna`
- Handle class imbalance with `class_weight="balanced"` or SMOTE
- Add SHAP values for per-customer explainability
- Calibrate probabilities (`CalibratedClassifierCV`) if scores will be used as
  actual default-probability estimates
