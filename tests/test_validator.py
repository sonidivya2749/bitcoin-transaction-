import pandas as pd
import pytest

from src.ingestion.validator import (
    _parse_list,
    _validate_ip,
    _validate_positive_number,
    filter_ledger_rows,
    filter_network_rows,
    validate_cross_file,
    validate_ledger,
    validate_network_log,
)

def valid_txid(number="1"):
    # Return a valid 64-character hexadecimal TXID.
    return number.zfill(64)


def valid_ledger_row():
    # Return one valid ledger record.
    return {
        "txid": valid_txid("1"),
        "timestamp": "2025-01-01 12:00:00",
        "input_addresses": "['1BoatSLRHtKNngkdXEeobR76b53LETtpyT', '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa']",
        "output_addresses": "['1dice8EMZmqKvrGE4Qc9bUFf9PX3xaYDp']", 
        "input_amounts": "[1.0, 2.0]",
        "output_amounts": "[2.9]",
        "fee": 0.1,
        "script_type": "P2PKH",
    }


def valid_network_row():
    # Return one valid network-log record.
    return {
        "txid": valid_txid("1"),
        "timestamp": "2025-01-01 12:00:00",
        "src_ip": "8.8.8.8",
        "dst_ip": "1.1.1.1",
        "src_port": 8333,
        "dst_port": 8333,
        "geo_country": "US",
        "asn": "AS15169",
    }


def test_parse_list_accepts_list():
    # Verify list values are accepted directly.
    value = ["addr1", "addr2"]

    assert _parse_list(value, "input_addresses", 0) == value


def test_parse_list_parses_string_list():
    # Verify string representations of lists are parsed.
    value = "['addr1', 'addr2']"

    assert _parse_list(value, "input_addresses", 0) == ["addr1","addr2",]


def test_parse_list_rejects_invalid_list():
    # Verify malformed list values are rejected.
    with pytest.raises(ValueError, match="invalid list"):
        _parse_list("NOT_A_LIST", "input_addresses", 0)


def test_parse_list_rejects_non_list_value():
    # Verify non-list values are rejected.
    with pytest.raises(ValueError, match="expected a list"):
        _parse_list("123", "input_addresses", 0)


def test_validate_positive_number_accepts_valid_value():
    # Verify positive finite numbers are accepted.
    assert _validate_positive_number( 1.5, "amount", 0,) is None


def test_validate_positive_number_rejects_negative_value():
    # Verify negative numbers are rejected.
    with pytest.raises(ValueError, match="invalid value"):
        _validate_positive_number(-1, "amount", 0,)


def test_validate_positive_number_rejects_zero():
    # Verify zero is rejected.
    with pytest.raises(ValueError, match="invalid value"):
        _validate_positive_number(0,"amount", 0,)


def test_validate_ip_accepts_global_ip():
    # Verify globally routable IP addresses are accepted.
    assert _validate_ip("8.8.8.8", "src_ip",0,) is None


def test_validate_ip_rejects_invalid_ip():
    # Verify malformed IP addresses are rejected.
    with pytest.raises(ValueError, match="invalid IP"):
        _validate_ip("999.999.999.999","src_ip", 0,)


def test_validate_ip_rejects_private_ip():
    # Verify private IP addresses are rejected.
    with pytest.raises(
        ValueError,
        match="not globally routable",
    ):
        _validate_ip("192.168.1.1", "src_ip", 0,)


def test_validate_ledger_accepts_valid_data():
    # Verify a valid ledger dataset passes validation.
    df = pd.DataFrame([valid_ledger_row()])

    result = validate_ledger(df)

    assert len(result) == 1
    assert isinstance(result.iloc[0]["input_addresses"], list)
    assert isinstance(result.iloc[0]["output_addresses"], list)
    assert isinstance(result.iloc[0]["input_amounts"], list)
    assert isinstance(result.iloc[0]["output_amounts"], list)
    assert result.iloc[0]["fee"] == 0.1


def test_validate_ledger_rejects_malformed_txid():
    # Verify malformed TXIDs are rejected.
    row = valid_ledger_row()
    row["txid"] = "INVALID_TXID"

    with pytest.raises(ValueError, match="malformed txid"):
        validate_ledger(pd.DataFrame([row]))


def test_validate_ledger_rejects_duplicate_txid():
    # Verify duplicate TXIDs are rejected.
    row1 = valid_ledger_row()
    row2 = valid_ledger_row()

    df = pd.DataFrame([row1, row2])

    with pytest.raises(ValueError, match="duplicate txid"):
        validate_ledger(df)


def test_validate_ledger_rejects_invalid_timestamp():
    # Verify invalid timestamps are rejected.
    row = valid_ledger_row()
    row["timestamp"] = "NOT_A_TIMESTAMP"

    with pytest.raises(ValueError, match="invalid timestamp"):
        validate_ledger(pd.DataFrame([row]))


def test_validate_ledger_rejects_negative_fee():
    # Verify negative fees are rejected.
    row = valid_ledger_row()
    row["fee"] = -0.1

    with pytest.raises(ValueError, match="invalid value"):
        validate_ledger(pd.DataFrame([row]))


def test_validate_ledger_rejects_malformed_input_list():
    # Verify malformed input-address lists are rejected.
    row = valid_ledger_row()
    row["input_addresses"] = "NOT_A_LIST"

    with pytest.raises(ValueError, match="invalid list"):
        validate_ledger(pd.DataFrame([row]))


def test_validate_ledger_rejects_amount_length_mismatch():
    # Verify input and amount list lengths must match.
    row = valid_ledger_row()
    row["input_amounts"] = "[1.0]"

    with pytest.raises(
        ValueError,
        match="length mismatch",
    ):
        validate_ledger(pd.DataFrame([row]))


def test_validate_ledger_rejects_negative_output():
    # Verify negative output amounts are rejected.
    row = valid_ledger_row()
    row["output_amounts"] = "[-1.0]"

    with pytest.raises(ValueError, match="invalid value"):
        validate_ledger(pd.DataFrame([row]))


def test_validate_ledger_rejects_empty_inputs():
    # Verify empty input lists are rejected.
    row = valid_ledger_row()
    row["input_addresses"] = "[]"
    row["input_amounts"] = "[]"

    with pytest.raises(
        ValueError,
        match="empty input_addresses",
    ):
        validate_ledger(pd.DataFrame([row]))


def test_validate_network_accepts_valid_data():
    # Verify a valid network dataset passes validation.
    df = pd.DataFrame([valid_network_row()])

    result = validate_network_log(df)

    assert len(result) == 1
    assert result.iloc[0]["src_port"] == 8333
    assert result.iloc[0]["dst_port"] == 8333
    assert result.iloc[0]["src_ip"] == "8.8.8.8"


def test_validate_network_rejects_invalid_port():
    # Verify invalid ports are rejected.
    row = valid_network_row()
    row["src_port"] = 99999

    with pytest.raises(ValueError, match="src_port contains invalid port",):
        validate_network_log(pd.DataFrame([row]))


def test_validate_network_rejects_invalid_ip():
    # Verify invalid network IPs are rejected.
    row = valid_network_row()
    row["src_ip"] = "999.999.999.999"

    with pytest.raises(ValueError, match="invalid IP"):
        validate_network_log(pd.DataFrame([row]))


def test_validate_network_rejects_private_ip():
    # Verify private network IPs are rejected.
    row = valid_network_row()
    row["src_ip"] = "192.168.1.1"

    with pytest.raises(
        ValueError,
        match="not globally routable",
    ):
        validate_network_log(pd.DataFrame([row]))


def test_validate_network_rejects_missing_asn():
    # Verify missing ASN values are rejected.
    row = valid_network_row()
    row["asn"] = None

    with pytest.raises(
        ValueError,
        match="null value",
    ):
        validate_network_log(pd.DataFrame([row]))


def test_validate_cross_file_accepts_matching_txids():
    # Verify matching TXIDs pass cross-file validation.
    ledger = pd.DataFrame(
        {
            "txid": [valid_txid("1")],
            "timestamp": [pd.Timestamp("2025-01-01 12:00:00")],
        }
    )
    network = pd.DataFrame(
        {
            "txid": [valid_txid("1")],
            "timestamp": [pd.Timestamp("2025-01-01 12:00:01")],
            "src_ip": ["8.8.8.8"],
            "dst_ip": ["1.1.1.1"],
        }
    )

    assert validate_cross_file(
        ledger,
        network,
    ) is True

def test_validate_cross_file_rejects_orphan_network_txid():
    # Verify network-only TXIDs are rejected.
    ledger = pd.DataFrame({"txid": [valid_txid("1")]})
    network = pd.DataFrame({"txid": [valid_txid("2")]})

    with pytest.raises(ValueError, match="no matching ledger",
    ):
        validate_cross_file(
            ledger,
            network,
        )


def test_validate_cross_file_rejects_missing_network_observation():
    # Verify ledger transactions without network observations are rejected.
    ledger = pd.DataFrame({"txid": [valid_txid("1"), valid_txid("2")]})
    network = pd.DataFrame({"txid": [valid_txid("1")]})

    with pytest.raises(
        ValueError,
        match="no network observation",
    ):
        validate_cross_file(
            ledger,
            network,
        )

def test_filter_ledger_rows_separates_invalid_rows():
    # Verify invalid ledger rows are separated from clean rows.
    valid_row = valid_ledger_row()
    invalid_row = valid_ledger_row()
    invalid_row["txid"] = "INVALID_TXID"

    df = pd.DataFrame([valid_row, invalid_row])
    clean_df, rejected_df = filter_ledger_rows(df)

    assert len(clean_df) == 1
    assert len(rejected_df) == 1
    assert rejected_df.iloc[0]["_row_index"] == 1
    assert "invalid txid" in rejected_df.iloc[0][
        "_rejection_reason"
    ]


def test_filter_network_rows_separates_invalid_rows():
    # Verify invalid network rows are separated from clean rows.
    valid_row = valid_network_row()
    invalid_row = valid_network_row()
    invalid_row["src_port"] = 99999

    df = pd.DataFrame([valid_row, invalid_row])
    clean_df, rejected_df = filter_network_rows(df)

    assert len(clean_df) == 1
    assert len(rejected_df) == 1
    assert rejected_df.iloc[0]["_row_index"] == 1
    assert "invalid src_port" in rejected_df.iloc[0][
        "_rejection_reason"
    ]


def test_filter_network_rows_normalizes_valid_values():
    # Verify valid network values are normalized.
    row = valid_network_row()
    row["src_port"] = "8333"
    row["dst_port"] = "8333"
    row["src_ip"] = " 8.8.8.8 "
    row["dst_ip"] = " 1.1.1.1 "
    row["geo_country"] = " US "
    row["asn"] = " AS15169 "

    clean_df, rejected_df = filter_network_rows(pd.DataFrame([row]))

    assert len(rejected_df) == 0
    assert clean_df.iloc[0]["src_port"] == 8333
    assert clean_df.iloc[0]["dst_port"] == 8333
    assert clean_df.iloc[0]["src_ip"] == "8.8.8.8"
    assert clean_df.iloc[0]["dst_ip"] == "1.1.1.1"
    assert clean_df.iloc[0]["geo_country"] == "US"
    assert clean_df.iloc[0]["asn"] == "AS15169"