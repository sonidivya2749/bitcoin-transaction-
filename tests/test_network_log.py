import pytest

from src.generate.ledger import build_ledger
from src.generate.pools import _flatten
from src.generate.network_log import generate_network_log
from src.generate.config import PEERS_PER_TRANSACTION_RANGE


@pytest.fixture(scope="module")
def network_data():
    # Build ledger and network log data for tests.
    ledger_df, pools, extra_wallets = build_ledger()

    all_wallets = _flatten(pools)
    if extra_wallets:
        all_wallets.extend(extra_wallets)

    network_log_df = generate_network_log(ledger_df, all_wallets)

    return ledger_df, network_log_df


def test_generate_network_log(network_data):
    # Verify network log contains relay rows.
    _, network_log_df = network_data

    assert len(network_log_df) > 0


def test_network_log_required_columns(network_data):
    # Verify all required network columns are present.
    _, network_log_df = network_data

    required_columns = {
        "txid",
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "geo_country",
        "asn",
    }

    missing = required_columns - set(network_log_df.columns)

    assert not missing


def test_network_txids_exist_in_ledger(network_data):
    # Verify every network TXID exists in the ledger.
    ledger_df, network_log_df = network_data

    ledger_txids = set(ledger_df["txid"])
    network_txids = set(network_log_df["txid"])

    assert network_txids.issubset(ledger_txids)


def test_every_ledger_transaction_has_network_relay(network_data):
    # Verify every ledger transaction has at least one relay hop.
    ledger_df, network_log_df = network_data

    ledger_txids = set(ledger_df["txid"])
    network_txids = set(network_log_df["txid"])

    assert ledger_txids.issubset(network_txids)


def test_peer_count_per_transaction(network_data):
    # Verify relay hops stay within the configured peer range.
    _, network_log_df = network_data

    lo, hi = PEERS_PER_TRANSACTION_RANGE

    hop_counts = network_log_df.groupby("txid").size()

    assert hop_counts.between(lo, hi).all()