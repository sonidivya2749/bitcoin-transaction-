import ast
import random
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIRTY_DATA_DIR = PROJECT_ROOT / "data" / "dirty_data"
DIRTY_DATA_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
LEDGER_CORRUPTIONS = {
    "null_txid": 1,
    "invalid_txid": 1,
    "duplicate_txid": 1,
    "invalid_timestamp": 1,
    "negative_fee": 1,
    "malformed_input_addresses": 1,
    "amount_length_mismatch": 1,
    "negative_output_amount": 1,
}
NETWORK_CORRUPTIONS = {
    "null_txid": 1,
    "invalid_txid": 1,
    "invalid_timestamp": 1,
    "invalid_src_ip": 1,
    "invalid_dst_ip": 1,
    "invalid_src_port": 1,
    "invalid_dst_port": 1,
    "null_asn": 1,
}


def _pick_index(df, used_indices):
    """Return an unused random row index."""
    available = list(set(df.index) - set(used_indices))

    if not available:
        raise ValueError("Not enough rows available for corruption injection.")

    index = random.choice(available)
    used_indices.add(index)
    return index


def _corrupt_ledger(df):
    """Inject controlled invalid records into a ledger copy."""
    df = df.copy()
    used_indices = set()

    # 1. Missing TXID
    for _ in range(LEDGER_CORRUPTIONS["null_txid"]):
        idx = _pick_index(df, used_indices)
        df.at[idx, "txid"] = None

    # 2. Invalid TXID format
    for _ in range(LEDGER_CORRUPTIONS["invalid_txid"]):
        idx = _pick_index(df, used_indices)
        df.at[idx, "txid"] = "INVALID_TXID"

    # 3. Duplicate TXID
    for _ in range(LEDGER_CORRUPTIONS["duplicate_txid"]):
        idx = _pick_index(df, used_indices)

        if idx == df.index[0]:
            source_idx = df.index[1]
        else:
            source_idx = df.index[0]

        df.at[idx, "txid"] = df.at[source_idx, "txid"]

    # 4. Invalid timestamp
    for _ in range(LEDGER_CORRUPTIONS["invalid_timestamp"]):
        idx = _pick_index(df, used_indices)
        df.at[idx, "timestamp"] = "NOT_A_TIMESTAMP"

    # 5. Negative fee
    for _ in range(LEDGER_CORRUPTIONS["negative_fee"]):
        idx = _pick_index(df, used_indices)
        df.at[idx, "fee"] = -abs(float(df.at[idx, "fee"]))

    # 6. Malformed input-address list
    for _ in range(LEDGER_CORRUPTIONS["malformed_input_addresses"]):
        idx = _pick_index(df, used_indices)
        df.at[idx, "input_addresses"] = "NOT_A_LIST"

    # 7. Input/address amount length mismatch
    for _ in range(LEDGER_CORRUPTIONS["amount_length_mismatch"]):
        idx = _pick_index(df, used_indices)

        input_addresses = df.at[idx, "input_addresses"]

        if isinstance(input_addresses, str):
            try:
                parsed = ast.literal_eval(input_addresses)
            except (ValueError, SyntaxError):
                parsed = []

            if isinstance(parsed, list) and parsed:
                df.at[idx, "input_amounts"] = str([])
            else:
                df.at[idx, "input_amounts"] = str([1.0])
        else:
            df.at[idx, "input_amounts"] = str([])

    # 8. Negative output amount
    for _ in range(LEDGER_CORRUPTIONS["negative_output_amount"]):
        idx = _pick_index(df, used_indices)

        output_amounts = df.at[idx, "output_amounts"]

        if isinstance(output_amounts, str):
            try:
                amounts = ast.literal_eval(output_amounts)
            except (ValueError, SyntaxError):
                amounts = []
        else:
            amounts = output_amounts

        if isinstance(amounts, list) and amounts:
            amounts[0] = -abs(float(amounts[0]))
            df.at[idx, "output_amounts"] = str(amounts)
        else:
            df.at[idx, "output_amounts"] = str([-1.0])

    return df


def _corrupt_network_log(df):
    """Inject controlled invalid records into a network-log copy."""
    df = df.copy()
    used_indices = set()

    # 1. Missing TXID
    for _ in range(NETWORK_CORRUPTIONS["null_txid"]):
        idx = _pick_index(df, used_indices)
        df.at[idx, "txid"] = None

    # 2. Invalid TXID
    for _ in range(NETWORK_CORRUPTIONS["invalid_txid"]):
        idx = _pick_index(df, used_indices)
        df.at[idx, "txid"] = "INVALID_TXID"

    # 3. Invalid timestamp
    for _ in range(NETWORK_CORRUPTIONS["invalid_timestamp"]):
        idx = _pick_index(df, used_indices)
        df.at[idx, "timestamp"] = "NOT_A_TIMESTAMP"

    # 4. Invalid source IP
    for _ in range(NETWORK_CORRUPTIONS["invalid_src_ip"]):
        idx = _pick_index(df, used_indices)
        df.at[idx, "src_ip"] = "999.999.999.999"

    # 5. Invalid destination IP
    for _ in range(NETWORK_CORRUPTIONS["invalid_dst_ip"]):
        idx = _pick_index(df, used_indices)
        df.at[idx, "dst_ip"] = "NOT_AN_IP"

    # 6. Invalid source port
    for _ in range(NETWORK_CORRUPTIONS["invalid_src_port"]):
        idx = _pick_index(df, used_indices)
        df.at[idx, "src_port"] = 99999

    # 7. Invalid destination port
    for _ in range(NETWORK_CORRUPTIONS["invalid_dst_port"]):
        idx = _pick_index(df, used_indices)
        df.at[idx, "dst_port"] = -1

    # 8. Missing ASN
    for _ in range(NETWORK_CORRUPTIONS["null_asn"]):
        idx = _pick_index(df, used_indices)
        df.at[idx, "asn"] = None

    return df


def generate_dirty_datasets(
    ledger_path=None,
    network_log_path=None,
    output_dir=None,
):
    #Read clean generated datasets, inject controlled corruption, and write dirty CSV datasets.
    if ledger_path is None:
        ledger_path = PROJECT_ROOT / "data" / "clean" / "ledger.csv"

    if network_log_path is None:
        network_log_path = (
            PROJECT_ROOT / "data" / "clean" / "network_log.csv"
        )

    if output_dir is None:
        output_dir = DIRTY_DATA_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)

    ledger_df = pd.read_csv(ledger_path)
    network_log_df = pd.read_csv(network_log_path)

    if ledger_df.empty:
        raise ValueError("Clean ledger dataset is empty.")

    if network_log_df.empty:
        raise ValueError("Clean network log dataset is empty.")

    dirty_ledger = _corrupt_ledger(ledger_df)
    dirty_network_log = _corrupt_network_log(network_log_df)

    dirty_ledger_path = output_dir / "ledger_dirty.csv"
    dirty_network_path = output_dir / "network_log_dirty.csv"

    dirty_ledger.to_csv(
        dirty_ledger_path,
        index=False,
    )

    dirty_network_log.to_csv(
        dirty_network_path,
        index=False,
    )

    return dirty_ledger_path, dirty_network_path


if __name__ == "__main__":
    ledger_path, network_path = generate_dirty_datasets()

    print(f"Dirty ledger written to: {ledger_path}")
    print(f"Dirty network log written to: {network_path}")