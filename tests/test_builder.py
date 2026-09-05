import networkx as nx
import pandas as pd
import pytest

from src.graph.builder import (
    build_graph,
    validate_graph,
    graph_stats,
    save_graph,
    load_graph,
)


class Wallet:
    def __init__(self, address, wallet_id="w1"):
        self.address = address
        self.wallet_id = wallet_id
        self.script_type = "P2PKH"


@pytest.fixture
def small_data():
    wallets = [
        Wallet("wallet_a", "w1"),
        Wallet("wallet_b", "w2"),
    ]

    timestamp = pd.Timestamp("2025-01-01 12:00:00")

    ledger_df = pd.DataFrame(
        [
            {
                "txid": "a" * 64,
                "timestamp": timestamp,
                "input_addresses": ["wallet_a"],
                "output_addresses": ["wallet_b"],
                "input_amounts": [1.1],
                "output_amounts": [1.0],
                "fee": 0.1,
                "script_type": "P2PKH",
            }
        ]
    )

    network_log_df = pd.DataFrame(
        [
            {
                "txid": "a" * 64,
                "timestamp": timestamp + pd.Timedelta(seconds=10),
                "src_ip": "8.8.8.8",
                "dst_ip": "1.1.1.1",
                "src_port": 50000,
                "dst_port": 8333,
                "geo_country": "US",
                "asn": "AS15169",
            }
        ]
    )

    return ledger_df, network_log_df, wallets


def test_build_graph(small_data):
    ledger_df, network_log_df, wallets = small_data

    graph = build_graph(
        ledger_df,
        network_log_df,
        wallets,
    )

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == 5
    assert graph.number_of_edges() == 4


def test_graph_stats(small_data):
    ledger_df, network_log_df, wallets = small_data

    graph = build_graph(
        ledger_df,
        network_log_df,
        wallets,
    )

    stats = graph_stats(graph)

    assert stats["wallet_nodes"] == 2
    assert stats["transaction_nodes"] == 1
    assert stats["ip_nodes"] == 2
    assert stats["transaction_edges"] == 2
    assert stats["relay_edges"] == 2
    assert stats["total_nodes"] == 5
    assert stats["total_edges"] == 4


def test_transaction_node_and_edges(small_data):
    ledger_df, network_log_df, wallets = small_data

    graph = build_graph(
        ledger_df,
        network_log_df,
        wallets,
    )

    txid = ledger_df.iloc[0]["txid"]
    tx_node = f"tx:{txid}"

    assert tx_node in graph
    assert graph.nodes[tx_node]["node_type"] == "transaction"
    assert graph.nodes[tx_node]["txid"] == txid
    assert graph.nodes[tx_node]["fee"] == pytest.approx(0.1)

    input_edges = [
        attrs
        for _, _, attrs in graph.in_edges(tx_node, data=True)
        if attrs.get("relation") == "input"
    ]

    output_edges = [
        attrs
        for _, _, attrs in graph.out_edges(tx_node, data=True)
        if attrs.get("relation") == "output"
    ]

    assert len(input_edges) == 1
    assert len(output_edges) == 1
    assert input_edges[0]["amount"] == pytest.approx(1.1)
    assert output_edges[0]["amount"] == pytest.approx(1.0)


def test_transaction_fee_not_duplicated_on_edges(small_data):
    ledger_df, network_log_df, wallets = small_data

    graph = build_graph(
        ledger_df,
        network_log_df,
        wallets,
    )

    txid = ledger_df.iloc[0]["txid"]
    tx_node = f"tx:{txid}"

    assert graph.nodes[tx_node]["fee"] == pytest.approx(0.1)

    transaction_edges = [
        attrs
        for u, v, attrs in graph.edges(data=True)
        if attrs.get("edge_type") == "transaction"
        and (u == tx_node or v == tx_node)
    ]

    assert all("fee" not in attrs for attrs in transaction_edges)


def test_graph_has_no_ground_truth_leakage(small_data):
    ledger_df, network_log_df, wallets = small_data

    graph = build_graph(
        ledger_df,
        network_log_df,
        wallets,
    )

    forbidden_keys = {
        "wallet_id",
        "scenario",
        "scenario_id",
        "role",
        "ip_behavior",
    }

    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "wallet":
            assert not forbidden_keys.intersection(attrs)


def test_validate_graph(small_data):
    ledger_df, network_log_df, wallets = small_data

    graph = build_graph(
        ledger_df,
        network_log_df,
        wallets,
    )

    assert validate_graph(
        graph,
        ledger_df,
        network_log_df,
        wallets,
    ) is True


def test_save_and_load_graph(small_data, tmp_path):
    ledger_df, network_log_df, wallets = small_data

    graph = build_graph(
        ledger_df,
        network_log_df,
        wallets,
    )

    filepath = tmp_path / "test_graph.graphml"

    saved_path = save_graph(graph, filepath)

    assert saved_path == filepath
    assert filepath.exists()

    reloaded_graph = load_graph(
        filepath,
        ledger_df,
        network_log_df,
        wallets,
    )

    assert isinstance(reloaded_graph, nx.MultiDiGraph)
    assert reloaded_graph.number_of_nodes() == graph.number_of_nodes()
    assert reloaded_graph.number_of_edges() == graph.number_of_edges()


def test_build_graph_rejects_unknown_wallet(small_data):
    ledger_df, network_log_df, wallets = small_data

    ledger_df = ledger_df.copy()
    ledger_df.at[0, "output_addresses"] = ["unknown_wallet"]

    with pytest.raises(
        ValueError,
        match="not a known wallet node",
    ):
        build_graph(
            ledger_df,
            network_log_df,
            wallets,
        )


def test_build_graph_rejects_amount_length_mismatch(small_data):
    ledger_df, network_log_df, wallets = small_data

    ledger_df = ledger_df.copy()
    ledger_df.at[0, "output_amounts"] = [1.0, 0.5]

    with pytest.raises(
        ValueError,
        match="output_addresses/output_amounts length mismatch",
    ):
        build_graph(
            ledger_df,
            network_log_df,
            wallets,
        )


def test_build_graph_rejects_relay_timestamp_before_transaction(small_data):
    ledger_df, network_log_df, wallets = small_data

    network_log_df = network_log_df.copy()
    network_log_df.at[0, "timestamp"] = ledger_df.at[0, "timestamp"]

    with pytest.raises(
        ValueError,
        match="not after the ledger transaction",
    ):
        build_graph(
            ledger_df,
            network_log_df,
            wallets,
        )


def test_build_graph_rejects_broken_relay_chain(small_data):
    ledger_df, network_log_df, wallets = small_data

    network_log_df = pd.concat(
        [
            network_log_df,
            pd.DataFrame(
                [
                    {
                        "txid": "a" * 64,
                        "timestamp": (
                            ledger_df.at[0, "timestamp"]
                            + pd.Timedelta(seconds=20)
                        ),
                        "src_ip": "9.9.9.9",
                        "dst_ip": "4.4.4.4",
                        "src_port": 50001,
                        "dst_port": 8333,
                        "geo_country": "US",
                        "asn": "AS15169",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    with pytest.warns(UserWarning, match="Network relay chain broken"):
        G = build_graph(
            ledger_df,
            network_log_df,
            wallets,
        )

    assert G.has_edge("9.9.9.9", "4.4.4.4")