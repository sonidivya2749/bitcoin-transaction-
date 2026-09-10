import os
import sys
import ast
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import load_ledger 

FEATURE_COLS = ["tx_count", "avg_amount", "total_amount", "avg_fee", "avg_time_gap_seconds"]

FEATURE_LABELS = {
    "tx_count": "transaction count",
    "avg_amount": "average transaction amount",
    "total_amount": "total transaction volume",
    "avg_fee": "average fee",
    "avg_time_gap_seconds": "time gap between transactions",
}


def _parse_list_cell(cell):
    if isinstance(cell, list):
        return cell
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError, TypeError):
        return []


def save_upload_to_temp(uploaded_file):
    
    suffix = Path(uploaded_file.name).suffix.lower()
    upload_dir = PROJECT_ROOT / "data" / "tmp_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = upload_dir / f"upload_{next(tempfile._get_candidate_names())}{suffix}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path


def parse_ledger(uploaded_file):
    tmp_path = save_upload_to_temp(uploaded_file)
    try:
        df = load_ledger(filepath=tmp_path)  
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in ["input_addresses", "output_addresses", "input_amounts", "output_amounts"]:
        df[col] = df[col].apply(_parse_list_cell)
    return df


def extract_features(ledger_df):
    records = []
    for _, row in ledger_df.iterrows():
        txid = row["txid"]
        ts = row["timestamp"]
        fee = row["fee"]
        for addr, amt in zip(row["input_addresses"], row["input_amounts"]):
            records.append({"wallet": addr, "txid": txid, "timestamp": ts, "amount": amt, "fee": fee, "role": "sender"})
        for addr, amt in zip(row["output_addresses"], row["output_amounts"]):
            records.append({"wallet": addr, "txid": txid, "timestamp": ts, "amount": amt, "fee": fee, "role": "receiver"})

    activity = pd.DataFrame(records).sort_values(["wallet", "timestamp"])

    features = activity.groupby("wallet").agg(
        tx_count=("txid", "count"),
        avg_amount=("amount", "mean"),
        total_amount=("amount", "sum"),
        avg_fee=("fee", "mean"),
        sender_count=("role", lambda x: (x == "sender").sum()),
        receiver_count=("role", lambda x: (x == "receiver").sum()),
    ).reset_index()

    def avg_gap(group):
        times = group["timestamp"].sort_values()
        if len(times) < 2:
            return 0.0
        return times.diff().dropna().dt.total_seconds().mean()

    gaps = activity.groupby("wallet").apply(avg_gap, include_groups=False).reset_index()
    gaps.columns = ["wallet", "avg_time_gap_seconds"]

    features = features.merge(gaps, on="wallet", how="left")
    features["avg_time_gap_seconds"] = features["avg_time_gap_seconds"].fillna(0.0)
    features["fan_ratio"] = features["sender_count"] / (features["receiver_count"] + 1)
    return features


def load_model(model_path):
    return joblib.load(model_path)


def score_wallets(features_df, model):
    X = features_df[FEATURE_COLS].values
    raw = -model.decision_function(X)
    lo, hi = raw.min(), raw.max()
    suspicion = (raw - lo) / (hi - lo) * 100 if hi > lo else np.zeros_like(raw)

    df = features_df.copy()
    df["suspicion_score"] = suspicion.round(2)
    df["is_anomaly"] = np.where(model.predict(X) == -1, "Yes", "No")
    df["confidence_pct"] = df["suspicion_score"]
    df["risk_level"] = pd.cut(df["suspicion_score"], bins=[-1, 40, 70, 101], labels=["Low", "Medium", "High"])
    return df


def explain_wallets(scored_df, model):
    import shap
    explainer = shap.TreeExplainer(model)
    values = explainer(scored_df[FEATURE_COLS]).values

    reasons = []
    for i in range(len(scored_df)):
        row_vals = values[i]
        top_idx = np.argsort(np.abs(row_vals))[::-1][:2]
        parts = []
        for j in top_idx:
            fname = FEATURE_LABELS[FEATURE_COLS[j]]
            direction = "pushed toward suspicious" if row_vals[j] < 0 else "pushed toward normal"
            parts.append(f"{fname} ({direction})")
        reasons.append("SHAP explanation: " + " ; ".join(parts))

    df = scored_df.copy()
    df["reason"] = np.where(df["is_anomaly"] == "Yes", reasons, "Normal behaviour -- no red flags")
    df = df.sort_values("suspicion_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def run_full_pipeline(uploaded_file, model_path, status_callback=None):
    def log(msg):
        if status_callback:
            status_callback(msg)

    log("Validating & ingesting uploaded file (CSV/JSON/XML) ...")
    ledger_df = parse_ledger(uploaded_file)
    log(f"Ingested and validated {len(ledger_df)} transactions.")

    log("Extracting per-wallet features ...")
    features_df = extract_features(ledger_df)
    log(f"Built features for {len(features_df)} wallets.")

    log("Loading trained Isolation Forest model ...")
    model = load_model(model_path)

    log("Scoring wallets for anomalies ...")
    scored_df = score_wallets(features_df, model)
    n_flagged = int((scored_df["is_anomaly"] == "Yes").sum())
    log(f"Flagged {n_flagged} wallets as anomalous.")

    log("Generating SHAP explanations ...")
    final_df = explain_wallets(scored_df, model)
    log("Done.")

    return final_df
