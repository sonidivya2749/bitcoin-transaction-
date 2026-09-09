"""
step6_clustering.py -- Louvain community detection on the transaction graph.

Idea: Wallets that transact heavily with each other form a "community".
In money-laundering, mixing, and peeling scenarios, the wallets involved
are usually tightly connected to each other -- Louvain clustering surfaces
these groups automatically, without needing to know the pattern in advance.

Output: wallet_clusters.csv  (wallet, cluster_id, cluster_size, cluster_risk)

Run from inside the models/ folder:
    python step6_clustering.py
"""

import os
import pandas as pd
import networkx as nx
import community as community_louvain  # pip install python-louvain

GRAPH_PATH = os.path.join("..", "data", "graph", "transaction_network.graphml")
SCORES_FILE = "wallet_scores.csv"
OUTPUT_FILE = "wallet_clusters.csv"

print("Loading graph ...")
G = nx.read_graphml(GRAPH_PATH)
print(f"  Total nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

# Louvain only works on undirected graphs
print("Converting to undirected ...")
G_undirected = G.to_undirected()

print("Running Louvain community detection (this can take a bit on large graphs) ...")
partition = community_louvain.best_partition(G_undirected)

cluster_df = pd.DataFrame(list(partition.items()), columns=["node", "cluster_id"])
print(f"  Found {cluster_df['cluster_id'].nunique()} total clusters (includes wallet, tx, and IP nodes).")

# --- Keep only wallet nodes (drop transaction/IP nodes from the cluster view) ---
node_types = nx.get_node_attributes(G, "node_type")
cluster_df["node_type"] = cluster_df["node"].map(node_types)
wallet_clusters = cluster_df[cluster_df["node_type"] == "wallet"].copy()
wallet_clusters = wallet_clusters.rename(columns={"node": "wallet"})[["wallet", "cluster_id"]]

print(f"  {len(wallet_clusters)} wallet nodes assigned to clusters.")

# --- Cluster size ---
cluster_sizes = wallet_clusters.groupby("cluster_id")["wallet"].transform("count")
wallet_clusters["cluster_size"] = cluster_sizes

# --- Bring in suspicion scores to compute a "cluster risk" score ---
if os.path.exists(SCORES_FILE):
    scores = pd.read_csv(SCORES_FILE)[["wallet", "suspicion_score", "is_anomaly"]]
    wallet_clusters = wallet_clusters.merge(scores, on="wallet", how="left")

    # cluster_risk = average suspicion score of all wallets in that cluster
    cluster_risk = wallet_clusters.groupby("cluster_id")["suspicion_score"].transform("mean")
    wallet_clusters["cluster_risk_score"] = cluster_risk.round(2)

    # how many anomalous wallets are in each cluster
    anomaly_count = wallet_clusters.groupby("cluster_id")["is_anomaly"].transform(lambda x: (x == "Yes").sum())
    wallet_clusters["cluster_anomaly_count"] = anomaly_count

wallet_clusters = wallet_clusters.sort_values(
    ["cluster_risk_score", "cluster_size"], ascending=[False, False]
).reset_index(drop=True)

wallet_clusters.to_csv(OUTPUT_FILE, index=False)
print(f"\n>> '{OUTPUT_FILE}' saved.\n")

# --- Show the riskiest clusters (highest avg suspicion, more than 1 wallet) ---
print("Top 10 riskiest clusters (size > 1):")
summary = wallet_clusters[wallet_clusters["cluster_size"] > 1].drop_duplicates("cluster_id")
print(summary[["cluster_id", "cluster_size", "cluster_risk_score", "cluster_anomaly_count"]].head(10).to_string(index=False))
