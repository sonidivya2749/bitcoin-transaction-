from pathlib import Path
import networkx as nx
import warnings
from collections import namedtuple
from ..ingestion.loader import load_ledger, load_wallets, load_network_log, validate_ingested_data
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_OUTPUT_DIR = PROJECT_ROOT / "data" / "graph"
GRAPH_OUTPUT_PATH = GRAPH_OUTPUT_DIR / "transaction_network.graphml"


def build_graph(ledger_df, network_log_df, all_wallets):
    G = nx.MultiDiGraph()
    _add_wallet_nodes(G, all_wallets)
    _add_transaction_edges(G, ledger_df)
    _add_ip_nodes_and_edges(G, ledger_df, network_log_df)
    validate_graph(G, ledger_df, network_log_df, all_wallets)
    return G


def _add_wallet_nodes(G, all_wallets):
    for wallet in all_wallets:
        if wallet.address in G:
            raise ValueError(
                f"Duplicate graph node address encountered: {wallet.address}"
            )
        G.add_node(wallet.address, node_type="wallet", script_type=wallet.script_type)


def _add_transaction_edges(G, ledger_df):
    for idx, row in ledger_df.iterrows():
        txid = row["txid"]
        tx_node = f"tx:{txid}"
        inputs, outputs = row["input_addresses"], row["output_addresses"]
        input_amounts, output_amounts = row["input_amounts"], row["output_amounts"]

        if len(inputs) != len(input_amounts):
            raise ValueError(
                f"Ledger row {idx}, txid {txid}: "
                "input_addresses/input_amounts length mismatch"
            )
        if len(outputs) != len(output_amounts):
            raise ValueError(
                f"Ledger row {idx}, txid {txid}: "
                "output_addresses/output_amounts length mismatch"
            )

        for address in inputs:
            if address not in G:
                raise ValueError(
                    f"Ledger row {idx}, txid {txid}: input address "
                    f"{address!r} is not a known wallet node"
                )

        for address in outputs:
            if address not in G:
                raise ValueError(
                    f"Ledger row {idx}, txid {txid}: output address "
                    f"{address!r} is not a known wallet node"
                )

        G.add_node(
            tx_node,
            node_type="transaction",
            txid=txid,
            timestamp=row["timestamp"].isoformat(),
            fee=float(row["fee"]),
            script_type=str(row["script_type"]),
        )

        for address, amount in zip(inputs, input_amounts):
            G.add_edge(
                address, tx_node,
                edge_type="transaction",
                relation="input",
                txid=txid,
                amount=float(amount),
            )

        for address, amount in zip(outputs, output_amounts):
            G.add_edge(
                tx_node, address,
                edge_type="transaction",
                relation="output",
                txid=txid,
                amount=float(amount),
            )


def _add_ip_nodes_and_edges(G, ledger_df, network_log_df):
    tx_to_ledger_timestamp = {}

    for idx, row in ledger_df.iterrows():
        txid = row["txid"]
        tx_node = f"tx:{txid}"

        if not row["input_addresses"]:
            raise ValueError(
                f"Ledger row {idx}, txid {txid}: "
                "transaction has no input address"
            )
        if tx_node not in G:
            raise ValueError(
                f"Ledger row {idx}, txid {txid}: "
                "transaction node is missing"
            )

        tx_to_ledger_timestamp[txid] = row["timestamp"]

    for txid, tx_rows in network_log_df.groupby("txid", sort=False):
        tx_node = f"tx:{txid}"

        if txid not in tx_to_ledger_timestamp:
            raise ValueError(
                f"Network txid {txid} has no corresponding ledger transaction"
            )

        tx_rows = tx_rows.sort_values("timestamp")
        ledger_timestamp = tx_to_ledger_timestamp[txid]
        previous_dst_ip, first_hop = None, True

        for idx, row in tx_rows.iterrows():
            src_ip, dst_ip = row["src_ip"], row["dst_ip"]
            row_timestamp = row["timestamp"]

            if row_timestamp <= ledger_timestamp:
                raise ValueError(
                    f"Network log row for txid {txid}: relay timestamp "
                    f"{row_timestamp} is not after the ledger transaction's "
                    f"timestamp {ledger_timestamp}"
                )

            if not first_hop and src_ip != previous_dst_ip:
                warnings.warn(
                    f"Network relay chain broken for txid {txid} at row {idx}: "
                    f"previous destination IP {previous_dst_ip!r} does not match "
                    f"current source IP {src_ip!r} "
                    "(likely caused by a dropped relay hop during row filtering)"
                )

            if src_ip not in G:
                G.add_node(src_ip, node_type="ip")
            if dst_ip not in G:
                G.add_node(dst_ip, node_type="ip")

            if first_hop:
                G.add_edge(
                    tx_node, src_ip,
                    edge_type="relay",
                    relation="transaction_to_ip_derived",
                    txid=txid,
                    timestamp=row_timestamp.isoformat(),
                )
                first_hop = False

            G.add_edge(
                src_ip, dst_ip,
                edge_type="relay",
                relation="observed",
                txid=txid,
                timestamp=row_timestamp.isoformat(),
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=int(row["src_port"]),
                dst_port=int(row["dst_port"]),
                geo_country=str(row["geo_country"]),
                asn=str(row["asn"]),
            )

            previous_dst_ip = dst_ip


def validate_graph(G, ledger_df, network_log_df, all_wallets):
    if not isinstance(G, nx.MultiDiGraph):
        raise TypeError("Graph must be a networkx.MultiDiGraph")

    expected_wallet_addresses = {w.address for w in all_wallets}
    expected_transaction_nodes = {f"tx:{txid}" for txid in ledger_df["txid"]}

    wallet_nodes, transaction_nodes, ip_nodes = (
        {n for n, a in G.nodes(data=True) if a.get("node_type") == t}
        for t in ("wallet", "transaction", "ip")
    )

    if wallet_nodes != expected_wallet_addresses:
        missing = expected_wallet_addresses - wallet_nodes
        unexpected = wallet_nodes - expected_wallet_addresses
        raise ValueError(
            "Graph wallet-node mismatch. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

    if len(wallet_nodes) != len(all_wallets):
        raise ValueError(
            f"Expected {len(all_wallets)} wallet nodes, "
            f"found {len(wallet_nodes)}"
        )

    if transaction_nodes != expected_transaction_nodes:
        missing = expected_transaction_nodes - transaction_nodes
        unexpected = transaction_nodes - expected_transaction_nodes
        raise ValueError(
            "Graph transaction-node mismatch. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("node_type")
        if node_type not in {"wallet", "transaction", "ip"}:
            raise ValueError(
                f"Node {node!r} has invalid node_type={node_type!r}"
            )

    required_attrs = {"txid", "timestamp", "fee", "script_type"}
    for tx_node in transaction_nodes:
        missing = required_attrs - set(G.nodes[tx_node])
        if missing:
            raise ValueError(
                f"Transaction node {tx_node!r} missing attributes: {missing}"
            )

    transaction_edges = [
        (u, v, a)
        for u, v, a in G.edges(data=True)
        if a.get("edge_type") == "transaction"
    ]

    transaction_edges_by_txid = {}
    for edge in transaction_edges:
        transaction_edges_by_txid.setdefault(edge[2].get("txid"), []).append(edge)

    ledger_txids = set(ledger_df["txid"])
    graph_transaction_txids = {a["txid"] for _, _, a in transaction_edges}
    missing_transaction_txids = ledger_txids - graph_transaction_txids

    if missing_transaction_txids:
        raise ValueError(
            f"{len(missing_transaction_txids)} ledger txid(s) "
            "have no transaction edge in graph, "
            f"e.g. {list(missing_transaction_txids)[:5]}"
        )

    for idx, row in ledger_df.iterrows():
        txid = row["txid"]
        tx_edges = transaction_edges_by_txid.get(txid, [])
        input_edges = [e for e in tx_edges if e[2].get("relation") == "input"]
        output_edges = [e for e in tx_edges if e[2].get("relation") == "output"]

        if len(input_edges) != len(row["input_addresses"]):
            raise ValueError(
                f"Ledger row {idx}, txid {txid}: expected "
                f"{len(row['input_addresses'])} input edges, "
                f"found {len(input_edges)}"
            )

        if len(output_edges) != len(row["output_addresses"]):
            raise ValueError(
                f"Ledger row {idx}, txid {txid}: expected "
                f"{len(row['output_addresses'])} output edges, "
                f"found {len(output_edges)}"
            )

        actual = sorted(float(a["amount"]) for _, _, a in input_edges)
        expected = sorted(float(a) for a in row["input_amounts"])
        if actual != expected:
            raise ValueError(
                f"Ledger row {idx}, txid {txid}: "
                "input edge amounts do not match ledger"
            )

        actual = sorted(float(a["amount"]) for _, _, a in output_edges)
        expected = sorted(float(a) for a in row["output_amounts"])
        if actual != expected:
            raise ValueError(
                f"Ledger row {idx}, txid {txid}: "
                "output edge amounts do not match ledger"
            )

    for u, v, attrs in transaction_edges:
        relation = attrs.get("relation")

        if relation == "input":
            if G.nodes[u].get("node_type") != "wallet":
                raise ValueError(
                    f"Transaction input source {u!r} is not a wallet node"
                )
            if G.nodes[v].get("node_type") != "transaction":
                raise ValueError(
                    f"Transaction input destination {v!r} "
                    "is not a transaction node"
                )
        elif relation == "output":
            if G.nodes[u].get("node_type") != "transaction":
                raise ValueError(
                    f"Transaction output source {u!r} "
                    "is not a transaction node"
                )
            if G.nodes[v].get("node_type") != "wallet":
                raise ValueError(
                    f"Transaction output destination {v!r} is not a wallet node"
                )
        else:
            raise ValueError(
                f"Transaction edge {u!r}->{v!r} "
                f"has invalid relation={relation!r}"
            )

        missing = {"txid", "amount", "relation"} - set(attrs)
        if missing:
            raise ValueError(
                f"Transaction edge {u!r}->{v!r} missing attributes: {missing}"
            )

    relay_edges = [
        (u, v, a)
        for u, v, a in G.edges(data=True)
        if a.get("edge_type") == "relay"
    ]

    network_txids = set(network_log_df["txid"])
    graph_network_txids = {a["txid"] for _, _, a in relay_edges}
    missing_network_txids = network_txids - graph_network_txids

    if missing_network_txids:
        raise ValueError(
            f"{len(missing_network_txids)} network txid(s) "
            "have no relay edge in graph, "
            f"e.g. {list(missing_network_txids)[:5]}"
        )

    observed_edges_per_txid, derived_edges_per_txid = {}, {}
    for _, _, attrs in relay_edges:
        txid, relation = attrs["txid"], attrs.get("relation")
        if relation == "observed":
            observed_edges_per_txid[txid] = observed_edges_per_txid.get(txid, 0) + 1
        elif relation == "transaction_to_ip_derived":
            derived_edges_per_txid[txid] = derived_edges_per_txid.get(txid, 0) + 1

    network_rows_per_txid = network_log_df.groupby("txid").size().to_dict()

    for txid, expected_count in network_rows_per_txid.items():
        actual_count = observed_edges_per_txid.get(txid, 0)

        if actual_count != expected_count:
            raise ValueError(
                f"Txid {txid}: expected {expected_count} observed "
                "relay edges from network_log.csv, "
                f"found {actual_count}"
            )

        if derived_edges_per_txid.get(txid, 0) != 1:
            raise ValueError(
                f"Txid {txid}: expected exactly one "
                "transaction_to_ip_derived edge"
            )

    for u, v, attrs in relay_edges:
        source_type = G.nodes[u].get("node_type")
        destination_type = G.nodes[v].get("node_type")
        relation = attrs.get("relation")

        if (
            source_type == "transaction"
            and destination_type == "ip"
            and relation == "transaction_to_ip_derived"
        ):
            required = {"txid", "timestamp", "relation"}
        elif (
            source_type == "ip"
            and destination_type == "ip"
            and relation == "observed"
        ):
            required = {
                "txid", "timestamp", "relation",
                "src_ip", "dst_ip",
                "src_port", "dst_port",
                "geo_country", "asn",
            }
        else:
            raise ValueError(
                f"Invalid relay edge: {u!r} ({source_type}) -> "
                f"{v!r} ({destination_type}) with "
                f"relation={relation!r}"
            )

        missing = required - set(attrs)
        if missing:
            raise ValueError(
                f"Relay edge {u!r}->{v!r} missing attributes: {missing}"
            )

    for ip_node in ip_nodes:
        if G.degree(ip_node) == 0:
            raise ValueError(f"Orphan IP node found in graph: {ip_node}")

    ledger_without_network = ledger_txids - network_txids
    if ledger_without_network:
        raise ValueError(
            f"{len(ledger_without_network)} ledger transaction(s) "
            "have no network observations"
        )

    network_without_ledger = network_txids - ledger_txids
    if network_without_ledger:
        raise ValueError(
            f"{len(network_without_ledger)} network txid(s) "
            "have no ledger transaction"
        )

    return True


def graph_stats(G):
    counts = {
        "wallet_nodes": 0,
        "transaction_nodes": 0,
        "ip_nodes": 0,
        "transaction_edges": 0,
        "relay_edges": 0,
    }

    for _, attrs in G.nodes(data=True):
        node_type = attrs.get("node_type")
        if node_type == "wallet":
            counts["wallet_nodes"] += 1
        elif node_type == "transaction":
            counts["transaction_nodes"] += 1
        elif node_type == "ip":
            counts["ip_nodes"] += 1

    for _, _, attrs in G.edges(data=True):
        edge_type = attrs.get("edge_type")
        if edge_type == "transaction":
            counts["transaction_edges"] += 1
        elif edge_type == "relay":
            counts["relay_edges"] += 1

    return {
        **counts,
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
    }


def save_graph(G, filepath=None):
    if filepath is None:
        filepath = GRAPH_OUTPUT_PATH
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, filepath)
    return filepath


def load_graph(
    filepath=None,
    ledger_df=None,
    network_log_df=None,
    all_wallets=None,
):
    if filepath is None:
        filepath = GRAPH_OUTPUT_PATH

    G = nx.read_graphml(filepath, force_multigraph=True)

    if (
        ledger_df is not None
        and network_log_df is not None
        and all_wallets is not None
    ):
        validate_graph(G, ledger_df, network_log_df, all_wallets)

    return G


Wallet = namedtuple("Wallet", ["address", "script_type"])
