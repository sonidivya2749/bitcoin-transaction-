import ast

import pandas as pd

from src.generate.corrupt import (
    _corrupt_ledger,
    _corrupt_network_log,
    generate_dirty_datasets,
)


def test_corrupt_ledger_injects_invalid_records():
    # Verify controlled corruptions are injected into ledger data.
    clean_df = pd.DataFrame(
        {
            "txid": [f"tx{i:064d}" for i in range(20)],
            "timestamp": ["2025-01-01 00:00:00"] * 20,
            "fee": [0.001] * 20,
            "input_addresses": [str(["addr1", "addr2"])] * 20,
            "input_amounts": [str([1.0, 2.0])] * 20,
            "output_amounts": [str([2.9])] * 20,
        }
    )

    dirty_df = _corrupt_ledger(clean_df)

    assert len(dirty_df) == len(clean_df)

    assert dirty_df["txid"].isna().any()
    assert "INVALID_TXID" in dirty_df["txid"].values
    assert (dirty_df["fee"] < 0).any()
    assert "NOT_A_TIMESTAMP" in dirty_df["timestamp"].values
    assert "NOT_A_LIST" in dirty_df["input_addresses"].values


def test_corrupt_ledger_injects_amount_corruptions():
    # Verify ledger amount lists are intentionally corrupted.
    clean_df = pd.DataFrame(
        {
            "txid": [f"tx{i:064d}" for i in range(20)],
            "timestamp": ["2025-01-01 00:00:00"] * 20,
            "fee": [0.001] * 20,
            "input_addresses": [str(["addr1", "addr2"])] * 20,
            "input_amounts": [str([1.0, 2.0])] * 20,
            "output_amounts": [str([2.9])] * 20,
        }
    )

    dirty_df = _corrupt_ledger(clean_df)

    mismatch_found = False
    negative_output_found = False

    for _, row in dirty_df.iterrows():
        try:
            addresses = ast.literal_eval(row["input_addresses"])
            amounts = ast.literal_eval(row["input_amounts"])

            if isinstance(addresses, list) and isinstance(amounts, list):
                if len(addresses) != len(amounts):
                    mismatch_found = True
        except (ValueError, SyntaxError, TypeError):
            pass

        try:
            outputs = ast.literal_eval(row["output_amounts"])

            if isinstance(outputs, list):
                if any(float(amount) < 0 for amount in outputs):
                    negative_output_found = True
        except (ValueError, SyntaxError, TypeError):
            pass

    assert mismatch_found
    assert negative_output_found


def test_corrupt_network_log_injects_invalid_records():
    # Verify controlled corruptions are injected into network logs.
    clean_df = pd.DataFrame(
        {
            "txid": [f"tx{i:064d}" for i in range(20)],
            "timestamp": ["2025-01-01 00:00:00"] * 20,
            "src_ip": ["192.168.1.10"] * 20,
            "dst_ip": ["10.0.0.10"] * 20,
            "src_port": [8333] * 20,
            "dst_port": [8333] * 20,
            "asn": [12345] * 20,
        }
    )

    dirty_df = _corrupt_network_log(clean_df)

    assert len(dirty_df) == len(clean_df)

    assert dirty_df["txid"].isna().any()
    assert "INVALID_TXID" in dirty_df["txid"].values
    assert "NOT_A_TIMESTAMP" in dirty_df["timestamp"].values
    assert "999.999.999.999" in dirty_df["src_ip"].values
    assert "NOT_AN_IP" in dirty_df["dst_ip"].values
    assert (dirty_df["src_port"] == 99999).any()
    assert (dirty_df["dst_port"] == -1).any()
    assert dirty_df["asn"].isna().any()


def test_generate_dirty_datasets_creates_files(tmp_path):
    # Verify dirty datasets are generated and written to disk.
    ledger_df = pd.DataFrame(
        {
            "txid": [f"tx{i:064d}" for i in range(20)],
            "timestamp": ["2025-01-01 00:00:00"] * 20,
            "fee": [0.001] * 20,
            "input_addresses": [str(["addr1", "addr2"])] * 20,
            "input_amounts": [str([1.0, 2.0])] * 20,
            "output_amounts": [str([2.9])] * 20,
        }
    )

    network_df = pd.DataFrame(
        {
            "txid": [f"tx{i:064d}" for i in range(20)],
            "timestamp": ["2025-01-01 00:00:00"] * 20,
            "src_ip": ["192.168.1.10"] * 20,
            "dst_ip": ["10.0.0.10"] * 20,
            "src_port": [8333] * 20,
            "dst_port": [8333] * 20,
            "asn": [12345] * 20,
        }
    )

    ledger_path = tmp_path / "ledger.csv"
    network_path = tmp_path / "network_log.csv"
    output_dir = tmp_path / "dirty_data"

    ledger_df.to_csv(ledger_path, index=False)
    network_df.to_csv(network_path, index=False)

    dirty_ledger_path, dirty_network_path = generate_dirty_datasets(
        ledger_path=ledger_path,
        network_log_path=network_path,
        output_dir=output_dir,
    )

    assert dirty_ledger_path.exists()
    assert dirty_network_path.exists()

    dirty_ledger = pd.read_csv(dirty_ledger_path)
    dirty_network = pd.read_csv(dirty_network_path)

    assert len(dirty_ledger) == len(ledger_df)
    assert len(dirty_network) == len(network_df)