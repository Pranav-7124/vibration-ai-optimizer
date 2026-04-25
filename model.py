"""
model.py  Multi-Model Training Pipeline for Vibration Amplitude Prediction
Trains Random Forest, Gradient Boosting (XGBoost-style), and MLP Neural Network.
Saves the best model + scaler for API use. Outputs full metrics and feature importances.
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  Load Dataset 
df = pd.read_csv("vibration_data.csv")
print(f" Dataset loaded: {len(df)} samples")

FEATURES = ["frequency", "mass_ratio", "clearance", "location", "freq_ratio", "damping_ratio"]
TARGET   = "amplitude"

X = df[FEATURES].values
y = df[TARGET].values

#  Train / Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

#  Feature Scaling 
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

#  Model Definitions 
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42
    ),
    "MLP Neural Network": MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        learning_rate_init=0.001,
        early_stopping=True,
        random_state=42,
    ),
}

#  Train & Evaluate 
results = {}
best_model_name = None
best_r2 = -np.inf
best_model_obj = None

print("\n" + "="*55)
print(f"{'Model':<25} {'R':>8} {'MAE':>10} {'RMSE':>10}")
print("="*55)

for name, mdl in models.items():
    # RF/GB do NOT need scaled data; MLP does
    use_scaled = "Neural" in name
    Xtr = X_train_s if use_scaled else X_train
    Xte = X_test_s  if use_scaled else X_test

    mdl.fit(Xtr, y_train)
    preds = mdl.predict(Xte)

    r2   = r2_score(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results[name] = {"r2": r2, "mae": mae, "rmse": rmse, "model": mdl, "scaled": use_scaled}
    print(f"{name:<25} {r2:>8.4f} {mae:>10.4f} {rmse:>10.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model_obj = mdl

print("="*55)
print(f"\n Best model: {best_model_name}  (R = {best_r2:.4f})")

#  Save Artifacts 
joblib.dump(best_model_obj, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print(" model.pkl and scaler.pkl saved")

# Save metadata (for API)
meta = {
    "best_model": best_model_name,
    "features": FEATURES,
    "target": TARGET,
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "metrics": {
        name: {"r2": round(v["r2"], 4), "mae": round(v["mae"], 6), "rmse": round(v["rmse"], 6)}
        for name, v in results.items()
    },
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(" model_meta.json saved")

#  Feature Importance (RF or GB) 
fi_model = results.get("Random Forest", results.get("Gradient Boosting"))
if fi_model and hasattr(fi_model["model"], "feature_importances_"):
    importances = fi_model["model"].feature_importances_
    fi_dict = dict(zip(FEATURES, importances.tolist()))
    fi_sorted = dict(sorted(fi_dict.items(), key=lambda x: x[1], reverse=True))

    # Save feature importances as JSON
    with open("feature_importance.json", "w") as f:
        json.dump(fi_sorted, f, indent=2)
    print(" feature_importance.json saved")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(list(fi_sorted.keys()), list(fi_sorted.values()), color="#00d4ff")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importances  {best_model_name}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    print(" feature_importance.png saved")

print("\n Training complete!")