# -*- coding: utf-8 -*-
import json
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import numpy as np
import uvicorn
from pathlib import Path
import pandas as pd  # <-- додано
import os
THRESHOLD = float(os.getenv("IDS_THRESHOLD", "0.5"))
MODEL_PATH = Path("artifacts/model.joblib")
FEAT_PATH  = Path("artifacts/features.json")

app = FastAPI(title="TrafficFlowClassifier")

# завантажуємо модель і метадані
try:
    if not MODEL_PATH.exists() or not FEAT_PATH.exists():
        raise FileNotFoundError("Немає artifacts/model.joblib або artifacts/features.json. Спочатку запусти python train_baseline.py")

    model = joblib.load(MODEL_PATH)
    meta = json.load(open(FEAT_PATH, "r", encoding="utf-8"))
    FEATURES  = meta["features"]
    NUM_COLS  = set(meta.get("num_cols", []))
    CAT_COLS  = set(meta.get("cat_cols", []))
except Exception as e:
    raise RuntimeError(f"Не вдалося завантажити артефакти: {e}")

class FlowFeatures(BaseModel):
    values: Dict[str, Any]

@app.post("/predict")
def predict(payload: FlowFeatures):
    try:
        row = payload.values or {}
        xdict: Dict[str, Any] = {}

        # підготуємо значення з правильними типами під КОЖНУ фічу
        for f in FEATURES:
            if f in NUM_COLS:
                val = row.get(f, 0)
                try:
                    val = float(val)
                except Exception:
                    val = 0.0
                xdict[f] = val
            elif f in CAT_COLS:
                val = row.get(f, "NA")
                xdict[f] = "NA" if val is None else str(val)
            else:
                # якщо з якоїсь причини фіча не в NUM/CAT списках
                xdict[f] = "NA" if row.get(f) is None else str(row.get(f))

        # ВАЖЛИВО: передаємо DataFrame, бо ColumnTransformer адресує колонки за іменами
        X_df = pd.DataFrame([xdict])

        proba = model.predict_proba(X_df)[0, 1]
        pred = int(proba >= THRESHOLD)
        return {"prediction": pred, "probability": float(proba), "model": meta.get("best_model", "?"), "threshold": THRESHOLD}

    except Exception as e:
        # Повертаємо зрозуміле 400 з деталями, щоб легше дебажити
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
