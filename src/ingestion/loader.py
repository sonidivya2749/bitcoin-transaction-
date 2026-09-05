import json
from pathlib import Path

import pandas as pd
from defusedxml import ElementTree as ET

from .security import (
    validate_file_security,
    validate_row_limit,
    validate_json_depth,
    validate_xml_depth,
)
from .schema import LEDGER_COLUMNS, NETWORK_LOG_COLUMNS
from .validator import (
    validate_ledger,
    validate_network_log,
    validate_wallets,
    validate_cross_file,
    filter_ledger_rows,
    filter_network_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_FORMATS = {".csv", ".json", ".xml"}


def _load_raw_file(filepath):
    # Load CSV, JSON, or XML into a raw DataFrame.
    filepath = validate_file_security(filepath)

    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(filepath)
        validate_row_limit(len(df), filepath.name)
        return df

    if suffix == ".json":
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)

        validate_json_depth(data)

        if isinstance(data, list):
            df = pd.DataFrame(data)
            validate_row_limit(len(df), filepath.name)
            return df

        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    df = pd.DataFrame(value)
                    validate_row_limit(len(df), filepath.name)
                    return df

            # Single-record JSON object.
            df = pd.DataFrame([data])
            validate_row_limit(len(df), filepath.name)
            return df

        raise ValueError(
            f"Unsupported JSON structure: {type(data).__name__}"
        )

    if suffix == ".xml":
        tree = ET.parse(filepath)
        root = tree.getroot()

        validate_xml_depth(root)

        records = []

        for record in root:
            row = {}

            for child in record:
                row[child.tag] = child.text

            if row:
                records.append(row)

        if not records:
            raise ValueError(f"XML file contains no records: {filepath}")

        df = pd.DataFrame(records)
        validate_row_limit(len(df), filepath.name)
        return df

    raise ValueError(
        f"Unsupported file format '{suffix}'. "
        f"Supported formats: CSV, JSON, XML"
    )


def _detect_dataset_type(df):
     # Detect whether the loaded DataFrame is a ledger or network log from its required columns.
    columns = set(df.columns)

    if set(LEDGER_COLUMNS).issubset(columns):
        return "ledger"

    if set(NETWORK_LOG_COLUMNS).issubset(columns):
        return "network_log"

    raise ValueError(
        "Unable to identify dataset type. "
        "Input does not match the required ledger or network-log schema."
    )


def load_ledger(filepath=None):
      #Load a ledger from CSV, JSON, or XML, filter invalid rows, and validate the resulting clean dataset.

    if filepath is None:
        filepath = PROJECT_ROOT / "data" / "clean" / "ledger.csv"

    df = _load_raw_file(filepath)

    if _detect_dataset_type(df) != "ledger":
        raise ValueError(
            f"Expected ledger dataset, but input was not recognized as "
            f"ledger: {filepath}"
        )

    clean_df, _ = filter_ledger_rows(df)

    if clean_df.empty:
        raise ValueError(
            f"Ledger contains no valid rows after filtering: {filepath}"
        )

    return validate_ledger(clean_df)

def load_network_log(filepath=None):
     
    # Load a network log from CSV, JSON, or XML, filter invalid rows, and validate the resulting clean dataset.
     
    if filepath is None:
        filepath = PROJECT_ROOT / "data" / "clean" / "network_log.csv"

    df = _load_raw_file(filepath)

    if _detect_dataset_type(df) != "network_log":
        raise ValueError(
            f"Expected network-log dataset, but input was not recognized "
            f"as network log: {filepath}"
        )

    clean_df, _ = filter_network_rows(df)

    if clean_df.empty:
        raise ValueError(
            f"Network log contains no valid rows after filtering: {filepath}"
        )

    return validate_network_log(clean_df)


def load_wallets(filepath=None):
    # Load wallet metadata from CSV, JSON, or XML, and validate it.
    if filepath is None:
        filepath = PROJECT_ROOT / "data" / "clean" / "wallet_metadata.csv"

    df = _load_raw_file(filepath)

    return validate_wallets(df)

def validate_ingested_data(ledger_df, network_log_df):

     # Validate the already-loaded ledger and network log together.
    if ledger_df.empty:
        raise ValueError("Ingested ledger is empty")

    if network_log_df.empty:
        raise ValueError("Ingested network log is empty")

    return validate_cross_file(
        ledger_df,
        network_log_df,
    )