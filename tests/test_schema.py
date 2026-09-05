import pandas as pd
import pytest

from src.ingestion.schema import (
    LEDGER_COLUMNS,
    LEDGER_LIST_COLUMNS,
    LEDGER_SCHEMA,
    NETWORK_LOG_COLUMNS,
    NETWORK_LOG_SCHEMA,
    check_columns,
)

def test_ledger_schema_definition():
    # Verify the ledger schema definition.
    assert LEDGER_SCHEMA.name == "ledger"
    assert LEDGER_SCHEMA.required_columns == LEDGER_COLUMNS
    assert LEDGER_SCHEMA.list_columns == LEDGER_LIST_COLUMNS

def test_network_schema_definition():
    # Verify the network-log schema definition.
    assert NETWORK_LOG_SCHEMA.name == "network_log"
    assert NETWORK_LOG_SCHEMA.required_columns == NETWORK_LOG_COLUMNS
    assert NETWORK_LOG_SCHEMA.list_columns == ()

def test_ledger_columns_are_defined():
    # Verify all required ledger columns are present in the schema.
    expected_columns = {
        "txid",
        "timestamp",
        "input_addresses",
        "output_addresses",
        "input_amounts",
        "output_amounts",
        "fee",
        "script_type",
    }

    assert set(LEDGER_COLUMNS) == expected_columns


def test_network_log_columns_are_defined():
    # Verify all required network-log columns are present in the schema.
    expected_columns = {
        "txid",
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "geo_country",
        "asn",
    }

    assert set(NETWORK_LOG_COLUMNS) == expected_columns


def test_check_columns_accepts_valid_dataframe():
    # Verify valid DataFrames pass column validation.
    df = pd.DataFrame(columns=LEDGER_COLUMNS)

    assert check_columns(
        df,
        LEDGER_COLUMNS,
        "ledger",
    ) is None


def test_check_columns_rejects_missing_column():
    # Verify missing required columns raise an error.
    columns = list(LEDGER_COLUMNS)
    columns.remove("fee")

    df = pd.DataFrame(columns=columns)

    with pytest.raises(
        ValueError,
        match="missing required column",
    ):
        check_columns(df, LEDGER_COLUMNS,"ledger",)


def test_check_columns_reports_multiple_missing_columns():
    # Verify multiple missing columns are reported.
    columns = list(NETWORK_LOG_COLUMNS)
    columns.remove("asn")
    columns.remove("geo_country")

    df = pd.DataFrame(columns=columns)

    with pytest.raises(
        ValueError,
        match="asn",
    ):
        check_columns(
            df,
            NETWORK_LOG_COLUMNS,
            "network_log",
        )
        