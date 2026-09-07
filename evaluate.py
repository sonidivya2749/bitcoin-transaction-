import pandas as pd

scores = pd.read_csv("wallet_scores.csv")
truth = pd.read_csv("data/ground_truth/wallet_ground_truth.csv")


merged = scores.merge(truth, left_on="wallet", right_on="address", how="inner")

print(f"Matched {len(merged)} / {len(scores)} scored wallets to ground truth.\n")


merged["actually_suspicious"] = merged["role"] != "normal"
merged["model_flagged"] = merged["is_anomaly"] == "Yes"

print("Ground truth breakdown (role column):")
print(merged["role"].value_counts())
print()

print("Scenario breakdown:")
print(merged["scenario"].value_counts())
print()

# Confusion matrix components
tp = ((merged["model_flagged"]) & (merged["actually_suspicious"])).sum()
fp = ((merged["model_flagged"]) & (~merged["actually_suspicious"])).sum()
fn = ((~merged["model_flagged"]) & (merged["actually_suspicious"])).sum()
tn = ((~merged["model_flagged"]) & (~merged["actually_suspicious"])).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("=" * 50)
print("MODEL ACCURACY AGAINST GROUND TRUTH")
print("=" * 50)
print(f"True Positives  (correctly flagged suspicious): {tp}")
print(f"False Positives (wrongly flagged normal):        {fp}")
print(f"False Negatives (missed actual suspicious):      {fn}")
print(f"True Negatives  (correctly left as normal):      {tn}")
print()
print(f"Precision: {precision:.2%}  (of what we flagged, how much was really bad)")
print(f"Recall:    {recall:.2%}  (of all actually-bad wallets, how much we caught)")
print(f"F1 Score:  {f1:.2%}")
print()

# Which scenarios are we catching vs missing?
print("Detection rate by scenario:")
scenario_stats = merged.groupby("scenario").agg(
    total=("wallet", "count"),
    caught=("model_flagged", "sum")
)
scenario_stats["catch_rate"] = (scenario_stats["caught"] / scenario_stats["total"] * 100).round(1)
print(scenario_stats)

merged.to_csv("evaluation_results.csv", index=False)
print("\nFull merged results saved to evaluation_results.csv")
