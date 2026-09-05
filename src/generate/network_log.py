import random
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .config import (
    PEERS_PER_TRANSACTION_RANGE,
    RELAY_DELAY_RANGE,
)
from .geoip import GeoIPLookup
from .ip_profiles import random_public_ip


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _random_port():
    """Bitcoin's standard P2P port is 8333; occasionally use a non-standard port."""
    if random.random() < 0.85:
        return 8333

    return random.randint(1024, 65535)


def generate_network_log(ledger_df, all_wallets):

    #Creates a P2P network log for transactions and links each relay to its txid.
    address_to_wallet = {w.address: w for w in all_wallets}
    rows = []
    geoip = GeoIPLookup()

    try:
        for _, tx in ledger_df.iterrows():
            txid = tx["txid"]
            tx_timestamp = tx["timestamp"]

            source_address = tx["input_addresses"][0]
            source_wallet = address_to_wallet.get(source_address)

            peer_count = random.randint(*PEERS_PER_TRANSACTION_RANGE)

            if source_wallet is not None:
                current_ip = source_wallet.ip_fn()
            else:
                current_ip = random_public_ip()

            current_timestamp = tx_timestamp

            for _ in range(peer_count):
                relay_delay = random.uniform(*RELAY_DELAY_RANGE)
                current_timestamp = (
                    current_timestamp
                    + timedelta(seconds=relay_delay)
                )

                next_ip = random_public_ip()

                geo_data = geoip.lookup(current_ip)

                row = {
                    "txid": txid,
                    "timestamp": current_timestamp,
                    "src_ip": current_ip,
                    "dst_ip": next_ip,
                    "src_port": _random_port(),
                    "dst_port": _random_port(),
                    "geo_country": geo_data["geo_country"],
                    "asn": geo_data["asn"],
                }

                rows.append(row)

                current_ip = next_ip

    finally:
        geoip.close()

    network_log_df = pd.DataFrame(rows)
    network_log_df = (
        network_log_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return network_log_df


def write_network_log_output(network_log_df):
    #Write the network log CSV to data/clean/, alongside ledger.csv.
    clean_dir = PROJECT_ROOT / "data" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    network_log_df.to_csv(
        clean_dir / "network_log.csv",
        index=False,
    )