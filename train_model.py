"""
Heart Disease Detection - Model Training Pipeline
--------------------------------------------------
Loads the dataset, cleans it, trains several classifiers, compares them,
and persists the best model + scaler + evaluation artifacts to disk.

Run:
    python train_model.py
"""

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

DATA_PATH = "data/heart.csv"
MODELS_DIR = "models"
PLOTS_DIR = "plots"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Missing values before cleaning:\n{df.isna().sum()}")

# ---------------------------------------------------------------------------
# 2. Handle missing values (median imputation per column - robust to outliers)
# ---------------------------------------------------------------------------
for col in FEATURE_COLUMNS:
    if df[col].isna().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

print(f"\nMissing values after cleaning:\n{df.isna().sum().sum()} total")

# ---------------------------------------------------------------------------
# 3. Label encoding (categorical columns are already numeric-coded per the
#    UCI schema, but we demonstrate LabelEncoder for any object dtype column
#    to make the pipeline robust to raw/string categorical input as well)
# ---------------------------------------------------------------------------
label_encoders = {}
for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

if label_encoders:
    joblib.dump(label_encoders, f"{MODELS_DIR}/label_encoders.pkl")
    print(f"Label-encoded columns: {list(label_encoders.keys())}")
else:
    print("No categorical/object columns needed label encoding (dataset already numeric-coded).")

X = df[FEATURE_COLUMNS]
y = df["target"]

# ---------------------------------------------------------------------------
# 4. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# 5. Feature scaling
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 6. Train multiple models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9),
}

if HAS_XGBOOST:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss", random_state=42,
    )
else:
    # Gradient Boosting as a strong stand-in when xgboost isn't installed
    models["Gradient Boosting"] = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
    )

results = {}
trained_models = {}

print("\n" + "=" * 60)
print("TRAINING & EVALUATING MODELS")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results[name] = {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
    }
    trained_models[name] = model

    print(f"\n{name}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")
    print(f"  F1-score : {f1*100:.2f}%")

# ---------------------------------------------------------------------------
# 7. Pick best model (highest accuracy)
# ---------------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = trained_models[best_name]
best_preds = best_model.predict(X_test_scaled)

print("\n" + "=" * 60)
print(f"BEST MODEL: {best_name}  (Accuracy: {results[best_name]['accuracy']}%)")
print("=" * 60)

# ---------------------------------------------------------------------------
# 8. Classification report
# ---------------------------------------------------------------------------
report = classification_report(y_test, best_preds, target_names=["No Disease", "Disease"])
print("\nClassification Report:\n", report)

with open(f"{PLOTS_DIR}/classification_report.txt", "w") as f:
    f.write(f"Best Model: {best_name}\n\n")
    f.write(report)

# ---------------------------------------------------------------------------
# 9. Confusion matrix plot
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"],
)
plt.title(f"Confusion Matrix - {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=150)
plt.close()
print(f"\nSaved confusion matrix -> {PLOTS_DIR}/confusion_matrix.png")

# ---------------------------------------------------------------------------
# 10. Accuracy comparison bar chart
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 5.5))
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
colors = ["#6C63FF" if n != best_name else "#22C55E" for n in names]
bars = plt.bar(names, accs, color=colors)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 100)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{acc}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/accuracy_comparison.png", dpi=150)
plt.close()
print(f"Saved accuracy comparison -> {PLOTS_DIR}/accuracy_comparison.png")

# ---------------------------------------------------------------------------
# 11. Persist model + scaler + metadata
# ---------------------------------------------------------------------------
joblib.dump(best_model, f"{MODELS_DIR}/heart_model.pkl")
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")

metadata = {
    "best_model": best_name,
    "feature_columns": FEATURE_COLUMNS,
    "all_results": results,
    "has_xgboost": HAS_XGBOOST,
}
with open(f"{MODELS_DIR}/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved model  -> {MODELS_DIR}/heart_model.pkl")
print(f"Saved scaler -> {MODELS_DIR}/scaler.pkl")
print(f"Saved metadata -> {MODELS_DIR}/model_metadata.json")
print("\nDONE.")
