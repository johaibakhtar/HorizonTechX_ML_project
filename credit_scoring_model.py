"""
Credit Scoring Model
=====================
Predicts customer credit risk (Low / Medium / High) using financial history
and transaction behavior features.

Pipeline:
1. Data generation / loading
2. Data preprocessing (cleaning, encoding, scaling)
3. Feature engineering
4. Model training (Logistic Regression, Decision Tree, Random Forest, XGBoost)
5. Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
6. Model comparison & selection
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import json

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUT_DIR = "/home/claude/outputs"
import os
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DATA GENERATION
# ---------------------------------------------------------------------------
# NOTE: No real-world dataset was provided for this task, so a realistic
# synthetic dataset is generated here that mirrors the structure of typical
# credit bureau / bank transaction data. To use your own data, replace this
# section with: df = pd.read_csv("your_file.csv")
# and make sure the column names line up with FEATURE_COLUMNS below,
# plus a target column named 'risk_category'.

def generate_credit_dataset(n_samples=6000):
    rng = np.random.default_rng(RANDOM_STATE)

    age = rng.integers(21, 65, n_samples)
    employment_years = np.clip(rng.normal(6, 4, n_samples), 0, 40)

    # Monthly income (right-skewed, like real income distributions)
    income = np.round(rng.lognormal(mean=10.5, sigma=0.5, size=n_samples), -2)
    income = np.clip(income, 8000, 500000)

    # Existing debts (loans, credit card balances owed)
    existing_debts = np.round(rng.lognormal(mean=9.5, sigma=1.0, size=n_samples), -2)
    existing_debts = np.clip(existing_debts, 0, income * 6)

    # Loan history: number of previous loans and number of past defaults
    num_previous_loans = rng.poisson(2.2, n_samples)
    num_previous_loans = np.clip(num_previous_loans, 0, 12)
    num_defaults = rng.binomial(num_previous_loans, p=np.clip(rng.beta(1.3, 9, n_samples), 0, 1))

    # Payment behavior: fraction of payments made on time (0-1) and avg days late
    on_time_payment_ratio = np.clip(rng.beta(6, 1.5, n_samples) - num_defaults * 0.03, 0, 1)
    avg_days_late = np.round(np.clip(rng.exponential(4, n_samples) + num_defaults * 3, 0, 90), 1)

    # Credit utilization: % of available credit currently used
    credit_utilization = np.clip(rng.beta(2, 3, n_samples) * 100 + num_defaults * 4, 0, 100)

    # Account balance (avg balance held in bank accounts)
    account_balance = np.round(np.clip(rng.lognormal(mean=8.5, sigma=1.2, size=n_samples), 0, None), -1)

    # Credit history length (years)
    credit_history_years = np.clip(rng.normal(7, 4, n_samples), 0, employment_years + 20)

    # Number of open credit lines
    num_open_credit_lines = rng.integers(0, 10, n_samples)

    df = pd.DataFrame({
        "age": age,
        "employment_years": np.round(employment_years, 1),
        "monthly_income": income,
        "existing_debts": existing_debts,
        "num_previous_loans": num_previous_loans,
        "num_defaults": num_defaults,
        "on_time_payment_ratio": np.round(on_time_payment_ratio, 3),
        "avg_days_late": avg_days_late,
        "credit_utilization_pct": np.round(credit_utilization, 1),
        "account_balance": account_balance,
        "credit_history_years": np.round(credit_history_years, 1),
        "num_open_credit_lines": num_open_credit_lines,
    })

    # --- Build a latent "risk score" to derive the ground-truth label ---
    debt_to_income = df["existing_debts"] / (df["monthly_income"] * 12)
    risk_score = (
        2.5 * debt_to_income
        + 0.03 * df["credit_utilization_pct"]
        + 1.4 * df["num_defaults"]
        - 1.8 * df["on_time_payment_ratio"]
        + 0.015 * df["avg_days_late"]
        - 0.05 * df["credit_history_years"]
        - 0.00002 * df["account_balance"]
        - 0.04 * df["employment_years"]
        + rng.normal(0, 0.6, n_samples)  # noise
    )

    low_thresh, high_thresh = np.percentile(risk_score, [40, 75])

    def bucket(score):
        if score <= low_thresh:
            return "Low Risk"
        elif score <= high_thresh:
            return "Medium Risk"
        else:
            return "High Risk"

    df["risk_category"] = risk_score.apply(bucket) if hasattr(risk_score, "apply") else \
        pd.Series(risk_score).apply(bucket)

    return df


df = generate_credit_dataset()
df.to_csv(f"{OUT_DIR}/credit_data.csv", index=False)
print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
print(df["risk_category"].value_counts())

# ---------------------------------------------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------------------------------------------
# Check for missing values / duplicates (synthetic data is clean, but this
# is where real-world cleaning would happen)
print("\nMissing values per column:\n", df.isnull().sum().sum())
df = df.drop_duplicates()

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
df["debt_to_income_ratio"] = df["existing_debts"] / (df["monthly_income"] * 12)
df["default_rate"] = df["num_defaults"] / df["num_previous_loans"].replace(0, 1)
df["balance_to_income_ratio"] = df["account_balance"] / df["monthly_income"]
df["credit_line_utilization_score"] = df["credit_utilization_pct"] * (1 - df["on_time_payment_ratio"])
df["is_new_credit_history"] = (df["credit_history_years"] < 2).astype(int)

FEATURE_COLUMNS = [
    "age", "employment_years", "monthly_income", "existing_debts",
    "num_previous_loans", "num_defaults", "on_time_payment_ratio",
    "avg_days_late", "credit_utilization_pct", "account_balance",
    "credit_history_years", "num_open_credit_lines",
    "debt_to_income_ratio", "default_rate", "balance_to_income_ratio",
    "credit_line_utilization_score", "is_new_credit_history",
]

X = df[FEATURE_COLUMNS].copy()
y = df["risk_category"].copy()

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # High=0, Low=1, Medium=2 (alphabetical) -- store mapping
class_names = le.classes_
print("\nClass mapping:", dict(zip(class_names, le.transform(class_names))))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=-1
    ),
}

results = {}
fitted_models = {}

for name, model in models.items():
    if name in ("Logistic Regression",):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    results[name] = {
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1-Score": f1, "ROC-AUC": roc_auc
    }
    fitted_models[name] = model

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print(f"ROC-AUC (macro, OVR): {roc_auc:.4f}")

# ---------------------------------------------------------------------------
# 5. MODEL COMPARISON
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results).T.round(4)
results_df = results_df.sort_values("F1-Score", ascending=False)
print("\n\nMODEL COMPARISON\n", results_df)
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv")

best_model_name = results_df.index[0]
best_model = fitted_models[best_model_name]
print(f"\nBest model: {best_model_name}")

joblib.dump(best_model, f"{OUT_DIR}/best_model_{best_model_name.replace(' ', '_')}.pkl")
joblib.dump(scaler, f"{OUT_DIR}/scaler.pkl")
joblib.dump(le, f"{OUT_DIR}/label_encoder.pkl")

with open(f"{OUT_DIR}/feature_columns.json", "w") as f:
    json.dump(FEATURE_COLUMNS, f)

# ---------------------------------------------------------------------------
# 6. VISUALIZATIONS
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")

# 6a. Model comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))
results_df[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].plot(
    kind="bar", ax=ax, colormap="viridis"
)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.legend(loc="lower right")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150)
plt.close()

# 6b. Confusion matrix for best model
if best_model_name == "Logistic Regression":
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title(f"Confusion Matrix - {best_model_name}", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# 6c. Feature importance (tree-based models)
if best_model_name in ("Random Forest", "Decision Tree", "XGBoost"):
    importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    importances.plot(kind="barh", ax=ax, color="teal")
    ax.set_title(f"Feature Importance - {best_model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
    plt.close()

# 6d. Class distribution
fig, ax = plt.subplots(figsize=(6, 5))
df["risk_category"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).plot(
    kind="bar", ax=ax, color=["#2ecc71", "#f39c12", "#e74c3c"]
)
ax.set_title("Risk Category Distribution", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/class_distribution.png", dpi=150)
plt.close()

print("\nAll artifacts saved to:", OUT_DIR)
