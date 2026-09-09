import pandas as pd
import numpy as np
import joblib
import shap

MODEL_FILE = "isolation_forest_model.pkl"
FEATURES_FILE = "wallet_features.csv"
SCORES_FILE = "wallet_scores.csv"
OUTPUT_FILE = "wallet_alerts_shap.csv"

FEATURE_COLS = ["tx_count", "avg_amount", "total_amount", "avg_fee", "avg_time_gap_seconds"]

FEATURE_LABELS = {
    "tx_count": "transaction count",
    "avg_amount": "average transaction amount",
    "total_amount": "total transaction volume",
    "avg_fee": "average fee",
    "avg_time_gap_seconds": "time gap between transactions",
}

print("Loading model and features ...")
model = joblib.load(MODEL_FILE)
features_df = pd.read_csv(FEATURES_FILE)
scores_df = pd.read_csv(SCORES_FILE)

X = features_df[FEATURE_COLS]

print("Building SHAP explainer (TreeExplainer) ...")
explainer = shap.TreeExplainer(model)

print("Computing SHAP values (this may take a bit for large datasets) ...")
shap_output = explainer(X)
values = shap_output.values  # shape: (n_wallets, n_features)

def readable_reason(i):
    row_vals = values[i]
    # Top 2 features by absolute SHAP contribution
    top_indices = np.argsort(np.abs(row_vals))[::-1][:2]
    parts = []
    for idx in top_indices:
        fname = FEATURE_LABELS[FEATURE_COLS[idx]]
        contribution = row_vals[idx]
        direction = "pushed toward suspicious" if contribution < 0 else "pushed toward normal"
        parts.append(f"{fname} ({direction}, impact={contribution:.3f})")
    return "SHAP explanation: " + " ; ".join(parts)

print("Generating human-readable reasons ...")
features_df["shap_reason"] = [readable_reason(i) for i in range(len(features_df))]

# Merge with scores (wallet, suspicion_score, is_anomaly)
merged = scores_df.merge(
    features_df[["wallet", "shap_reason"]], on="wallet", how="left"
)

def risk_band(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    return "Low"

merged["confidence_pct"] = merged["suspicion_score"].round(1)
merged["risk_level"] = merged["suspicion_score"].apply(risk_band)
merged["reason"] = np.where(
    merged["is_anomaly"] == "Yes",
    merged["shap_reason"],
    "Normal behaviour -- no red flags"
)

merged = merged.sort_values("suspicion_score", ascending=False).reset_index(drop=True)
merged.insert(0, "rank", merged.index + 1)

final_cols = ["rank", "wallet", "confidence_pct", "risk_level", "is_anomaly", "reason"] + FEATURE_COLS
merged = merged[final_cols]
merged.to_csv(OUTPUT_FILE, index=False)

n_alerts = int((merged["is_anomaly"] == "Yes").sum())
print(f"\n>> '{OUTPUT_FILE}' ban gaya -- {n_alerts} alerts with SHAP explanations.\n")
print("Top 5 alerts:")
print(merged[merged["is_anomaly"] == "Yes"][["rank", "wallet", "confidence_pct", "risk_level", "reason"]].head(5).to_string(index=False))
