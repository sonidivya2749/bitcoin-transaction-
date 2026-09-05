from dataclasses import dataclass


LEDGER_COLUMNS = (
    "txid",
    "timestamp",
    "input_addresses",
    "output_addresses",
    "input_amounts",
    "output_amounts",
    "fee",
    "script_type",
)

NETWORK_LOG_COLUMNS = (
    "txid",
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "geo_country",
    "asn",
)

WALLET_COLUMNS = (
    "address",
    "script_type",
)

LEDGER_LIST_COLUMNS = (
    "input_addresses",
    "output_addresses",
    "input_amounts",
    "output_amounts",
)


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    required_columns: tuple[str, ...]
    list_columns: tuple[str, ...] = ()


LEDGER_SCHEMA = DatasetSchema(
    name="ledger",
    required_columns=LEDGER_COLUMNS,
    list_columns=LEDGER_LIST_COLUMNS,
)

NETWORK_LOG_SCHEMA = DatasetSchema(
    name="network_log",
    required_columns=NETWORK_LOG_COLUMNS,
)


def check_columns(df, required_columns, dataset_name):
     
     # Ensure all required columns are present in the DataFrame.
     
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset_name}: missing required column(s): {sorted(missing)}"
        )