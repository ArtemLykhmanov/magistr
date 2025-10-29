# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# === НАЛАШТУВАННЯ ===
CSV_PATH = "data/UNSW_NB15_training-set.csv"  
LABEL_CANDIDATES = ["label", "Label", "class", "attack_cat", "target"]
TIMESTAMP_CANDIDATES = ["timestamp", "start_time", "flow_start", "StartTime", "stime", "ltime"]

MODEL_DIR = Path("artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# === ЗАВАНТАЖЕННЯ ДАНИХ ===
df = pd.read_csv(CSV_PATH)

# Визначаємо колонку-мітку
label_col = None
for c in LABEL_CANDIDATES:
    if c in df.columns:
        label_col = c
        break
if label_col is None:
    raise ValueError(f"Не знайшов колонку мітки серед {LABEL_CANDIDATES}")

# === Уніфікуємо бінарну мітку: 1=malicious, 0=benign ===
POS_LABELS = {
    "malicious","attack","anomaly","botnet","dos","ddos","malware",
    "infiltration","portscan","bruteforce","brute force","1","true"
}
y_raw = df[label_col]

if pd.api.types.is_bool_dtype(y_raw):
    y = y_raw.astype(int).values
elif pd.api.types.is_numeric_dtype(y_raw):
    y = (y_raw.astype(float) > 0).astype(int).values
else:
    y = (
        y_raw.astype(str)
             .str.strip()
             .str.lower()
             .isin(POS_LABELS)
             .astype(int)
             .values
    )
print("Label distribution:", pd.Series(y).value_counts().to_dict())

# === Прибираємо потенційні «витоки» і ID-поля зі фіч ===
LEAKY_COLS = [
    "attack_cat", "Attack_cat", "category", "subcategory",
    "label", "Label"
]
drop_cols = set([label_col]) | set([c for c in LEAKY_COLS if c in df.columns])
drop_cols |= set([c for c in ["srcIP","dstIP","srcPort","dstPort","DstIP","DstPort","Flow ID","flow_id","id","ID"]
                  if c in df.columns])

Xdf = df.drop(columns=list(drop_cols), errors="ignore")

# === Розділяємо на числові/категоріальні ===
num_cols = [c for c in Xdf.columns if pd.api.types.is_numeric_dtype(Xdf[c])]
cat_cols = [c for c in Xdf.columns if c not in num_cols]

# Мінімальне очищення
Xdf = Xdf.replace([np.inf, -np.inf], np.nan)
for c in num_cols:
    Xdf[c] = Xdf[c].fillna(Xdf[c].median())
for c in cat_cols:
    Xdf[c] = Xdf[c].fillna("NA")

print(f"Detected numeric features: {len(num_cols)} | categorical: {len(cat_cols)}")

# === Time-aware split якщо знайдемо часову колонку ===
ts_col = None
for c in TIMESTAMP_CANDIDATES:
    if c in df.columns:
        ts_col = c
        break

if ts_col:
    def to_dt(x):
        try:
            return pd.to_datetime(x)
        except Exception:
            try:
                return pd.to_datetime(float(x), unit="s")
            except Exception:
                return pd.NaT
    order = Xdf.assign(_ts=df[ts_col].apply(to_dt)).sort_values("_ts")
    Xdf = order.drop(columns=["_ts"])
    y = y[order.index.values]
    split_idx = int(0.8 * len(Xdf))
    X_train, X_test = Xdf.iloc[:split_idx], Xdf.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
else:
    X_train, X_test, y_train, y_test = train_test_split(
        Xdf, y, test_size=0.2, stratify=y, random_state=42
    )

# === Препроцесинг: скейл числових + one-hot для категоріальних ===
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", ohe, cat_cols),
    ],
    remainder="drop"
)

# === Моделі ===
logreg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

rf = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42
    ))
])

models = {"logreg": logreg, "rf": rf}
best_name, best_model, best_score = None, None, -1
results = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    pr_auc = average_precision_score(y_test, proba)
    f1 = f1_score(y_test, y_pred)
    results[name] = {"PR_AUC": pr_auc, "F1": f1}

    if pr_auc > best_score:
        best_name, best_model, best_score = name, pipe, pr_auc

print("=== РЕЗУЛЬТАТИ ===")
for k, v in results.items():
    print(f"{k}: PR-AUC={v['PR_AUC']:.4f} | F1={v['F1']:.4f}")

# Детальніший звіт для кращої моделі
best_proba = best_model.predict_proba(X_test)[:, 1]
best_pred = (best_proba >= 0.5).astype(int)
print("\n=== Classification report (best) ===")
print(classification_report(y_test, best_pred, digits=4))

# === Збереження артефактів ===
joblib.dump(best_model, MODEL_DIR / "model.joblib")

meta = {
    "features": num_cols + cat_cols,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "best_model": best_name
}
json.dump(meta, open(MODEL_DIR / "features.json", "w"), ensure_ascii=False, indent=2)

print("\nАртефакти збережено до:", MODEL_DIR.resolve())

