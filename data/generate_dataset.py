"""
Generates a synthetic dataset that follows the exact schema and realistic
statistical relationships of the classic UCI Cleveland Heart Disease dataset.

Columns (standard UCI Heart Disease schema):
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,
    slope, ca, thal, target

This script also randomly injects a small percentage of missing values so the
training pipeline demonstrates real missing-value handling.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000


def generate_row(has_disease):
    age = int(np.clip(np.random.normal(58 if has_disease else 50, 9), 29, 77))
    sex = np.random.choice([0, 1], p=[0.32, 0.68])  # 1 = male, 0 = female

    if has_disease:
        cp = np.random.choice([0, 1, 2, 3], p=[0.55, 0.15, 0.15, 0.15])
    else:
        cp = np.random.choice([0, 1, 2, 3], p=[0.15, 0.25, 0.30, 0.30])

    trestbps = int(np.clip(np.random.normal(134 if has_disease else 128, 17), 94, 200))
    chol = int(np.clip(np.random.normal(250 if has_disease else 235, 45), 126, 564))
    fbs = np.random.choice([0, 1], p=[0.85, 0.15])
    restecg = np.random.choice([0, 1, 2], p=[0.5, 0.45, 0.05])
    thalach = int(np.clip(np.random.normal(140 if has_disease else 158, 20), 71, 202))
    exang = np.random.choice([0, 1], p=[0.30, 0.70] if has_disease else [0.85, 0.15])
    oldpeak = round(float(np.clip(np.random.exponential(1.6 if has_disease else 0.6), 0, 6.2)), 1)
    slope = np.random.choice([0, 1, 2], p=[0.55, 0.35, 0.10] if has_disease else [0.15, 0.55, 0.30])
    ca = np.random.choice([0, 1, 2, 3], p=[0.35, 0.30, 0.20, 0.15] if has_disease else [0.75, 0.15, 0.07, 0.03])
    thal = np.random.choice([1, 2, 3], p=[0.05, 0.25, 0.70] if has_disease else [0.10, 0.75, 0.15])

    return [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]


rows = []
targets = []
for i in range(N):
    has_disease = 1 if i < N * 0.46 else 0
    rows.append(generate_row(has_disease))
    targets.append(has_disease)

df = pd.DataFrame(
    rows,
    columns=[
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    ],
)
df["target"] = targets
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Inject a small percentage of missing values to simulate real-world dirty data
for col in ["trestbps", "chol", "thalach", "ca", "thal"]:
    idx = df.sample(frac=0.02, random_state=np.random.randint(0, 10000)).index
    df.loc[idx, col] = np.nan

df.to_csv("/home/claude/heart_disease_system/data/heart.csv", index=False)
print("Dataset generated:", df.shape)
print(df["target"].value_counts())
print(df.isna().sum())
