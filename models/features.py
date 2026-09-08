"""
STEP: Extract per-wallet features from the real ledger data.

Input:  data/clean/ledger.csv
Output: wallet_features.csv  (one row per wallet address)

Run this from inside the repo folder:
    python step4_features.py
"""

import pandas as pd
import ast
from datetime import datetime

print("Loading ledger.csv ...")
df = pd.read_csv("data/clean/ledger.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# The address/amount columns are stored as strings that LOOK like Python lists,
# e.g. "['bc1q...', 'bc1q...']" -- we need to actually parse them into real lists.
def parse_list(cell):
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return []

print("Parsing address/amount columns ...")
df["input_addresses"] = df["input_addresses"].apply(parse_list)
df["output_addresses"] = df["output_addresses"].apply(parse_list)
df["input_amounts"] = df["input_amounts"].apply(parse_list)
df["output_amounts"] = df["output_amounts"].apply(parse_list)

# --- Build one row per (wallet, transaction, role, amount) ---
# A wallet can appear as a SENDER (input) or RECEIVER (output) of a transaction.
# We treat both roles as "activity" for that wallet.

records = []

for _, row in df.iterrows():
    txid = row["txid"]
    ts = row["timestamp"]
    fee = row["fee"]

    # Sender side
    for addr, amt in zip(row["input_addresses"], row["input_amounts"]):
        records.append({"wallet": addr, "txid": txid, "timestamp": ts, "amount": amt, "fee": fee, "role": "sender"})

    # Receiver side
    for addr, amt in zip(row["output_addresses"], row["output_amounts"]):
        records.append({"wallet": addr, "txid": txid, "timestamp": ts, "amount": amt, "fee": fee, "role": "receiver"})

print(f"Built {len(records)} wallet-transaction records, aggregating per wallet ...")
activity = pd.DataFrame(records)
activity = activity.sort_values(["wallet", "timestamp"])

# --- Aggregate features per wallet ---
features = activity.groupby("wallet").agg(
    tx_count=("txid", "count"),
    avg_amount=("amount", "mean"),
    total_amount=("amount", "sum"),
    avg_fee=("fee", "mean"),
    sender_count=("role", lambda x: (x == "sender").sum()),
    receiver_count=("role", lambda x: (x == "receiver").sum()),
).reset_index()

# --- Average time gap between a wallet's consecutive transactions ---
def avg_time_gap(group):
    times = group["timestamp"].sort_values()
    if len(times) < 2:
        return 0.0
    diffs = times.diff().dropna().dt.total_seconds()
    return diffs.mean()

time_gaps = activity.groupby("wallet").apply(avg_time_gap, include_groups=False).reset_index()
time_gaps.columns = ["wallet", "avg_time_gap_seconds"]

features = features.merge(time_gaps, on="wallet", how="left")
features["avg_time_gap_seconds"] = features["avg_time_gap_seconds"].fillna(0.0)

# Fan-in / fan-out ratio -- useful anomaly signal (peeling chains have very lopsided ratios)
features["fan_ratio"] = features["sender_count"] / (features["receiver_count"] + 1)

features.to_csv("wallet_features.csv", index=False)
print(f"Done! wallet_features.csv written with {len(features)} wallets.")
print(features.head(10))
