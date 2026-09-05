import pandas as pd
import pytest

from src.ingestion.loader import (
    _load_raw_file,
    load_ledger,
    load_network_log,
)

def test_load_csv(tmp_path):
    # Verify CSV files are loaded correctly.
    filepath = tmp_path / "data.csv"
    filepath.write_text(
        "txid,fee\n"
        "tx001,0.001\n"
        "tx002,0.002\n",
        encoding="utf-8",
    )

    df = _load_raw_file(filepath)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["txid", "fee"]


def test_load_json_list(tmp_path):
    # Verify JSON lists are loaded correctly.
    filepath = tmp_path / "data.json"
    filepath.write_text(
        '[{"txid": "tx001", "fee": 0.001}, '
        '{"txid": "tx002", "fee": 0.002}]',
        encoding="utf-8",
    )

    df = _load_raw_file(filepath)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "txid" in df.columns
    assert "fee" in df.columns


def test_load_json_wrapper(tmp_path):
    # Verify wrapped JSON records are loaded correctly.
    filepath = tmp_path / "data.json"
    filepath.write_text(
        '{"transactions": ['
        '{"txid": "tx001", "fee": 0.001}, '
        '{"txid": "tx002", "fee": 0.002}'
        ']}',
        encoding="utf-8",
    )

    df = _load_raw_file(filepath)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "txid" in df.columns


def test_load_single_record_json(tmp_path):
    # Verify a single JSON record is loaded correctly.
    filepath = tmp_path / "data.json"
    filepath.write_text(
        '{"txid": "tx001", "fee": 0.001}',
        encoding="utf-8",
    )

    df = _load_raw_file(filepath)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["txid"] == "tx001"


def test_load_xml(tmp_path):
    # Verify XML records are loaded correctly.
    filepath = tmp_path / "data.xml"
    filepath.write_text(
        "<root>"
        "<record>"
        "<txid>tx001</txid>"
        "<fee>0.001</fee>"
        "</record>"
        "<record>"
        "<txid>tx002</txid>"
        "<fee>0.002</fee>"
        "</record>"
        "</root>",
        encoding="utf-8",
    )

    df = _load_raw_file(filepath)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "txid" in df.columns
    assert "fee" in df.columns


def test_missing_file_is_rejected(tmp_path):
    # Verify missing input files are rejected.
    filepath = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Input file not found",
    ):
        _load_raw_file(filepath)


def test_unsupported_format_is_rejected(tmp_path):
    # Verify unsupported file formats are rejected.
    filepath = tmp_path / "data.txt"
    filepath.write_text("test", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Unsupported file format",
    ):
        _load_raw_file(filepath)


def test_empty_file_is_rejected(tmp_path):
    # Verify empty input files are rejected.
    filepath = tmp_path / "empty.csv"
    filepath.touch()

    with pytest.raises(
        ValueError,
        match="Input file is empty",
    ):
        _load_raw_file(filepath)


def test_invalid_json_structure_is_rejected(tmp_path):
    # Verify unsupported JSON structures are rejected.
    filepath = tmp_path / "data.json"
    filepath.write_text(
        '"invalid structure"',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported JSON structure",
    ):
        _load_raw_file(filepath)


def test_empty_xml_records_are_rejected(tmp_path):
    # Verify XML files without records are rejected.
    filepath = tmp_path / "empty.xml"
    filepath.write_text(
        "<root></root>",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="XML file contains no records",
    ):
        _load_raw_file(filepath)

def test_load_ledger_filters_invalid_rows(tmp_path):
    # Verify the production ledger loader filters dirty rows.
    dirty_ledger = pd.DataFrame([
        {
            "txid": "a" * 64,
            "timestamp": "2025-06-01 12:00:00",
            "input_addresses": ["1DG5ZNWddEt4USseTiz2CMPLWW1UT7mytT"],
            "output_addresses": ["15KCqiLdSHEou7okSG26LtjSJVTq7Dx2wX"],
            "input_amounts": [1.1],
            "output_amounts": [1.0],
            "fee": 0.1,
            "script_type": "P2PKH",
        },
        {
            "txid": "invalid_txid",
            "timestamp": "2025-06-01 12:00:00",
            "input_addresses": ["wallet_a"],
            "output_addresses": ["wallet_b"],
            "input_amounts": [1.1],
            "output_amounts": [1.0],
            "fee": 0.1,
            "script_type": "P2PKH",
        },
    ])

    filepath = tmp_path / "dirty_ledger.csv"
    dirty_ledger.to_csv(filepath, index=False)

    clean_df = load_ledger(filepath)

    assert len(clean_df) == 1
    assert clean_df.iloc[0]["txid"] == "a" * 64

def test_load_network_log_filters_invalid_rows(tmp_path):
    # Verify the production network loader filters dirty rows.
    dirty_network = pd.DataFrame([
        {
            "txid": "a" * 64,
            "timestamp": "2025-06-01 12:00:01",
            "src_ip": "8.8.8.8",
            "dst_ip": "1.1.1.1",
            "src_port": 50000,
            "dst_port": 8333,
            "geo_country": "US",
            "asn": "AS15169",
        },
        {
            "txid": "invalid_txid",
            "timestamp": "2025-06-01 12:00:01",
            "src_ip": "8.8.8.8",
            "dst_ip": "1.1.1.1",
            "src_port": 50000,
            "dst_port": 8333,
            "geo_country": "US",
            "asn": "AS15169",
        },
    ])

    filepath = tmp_path / "dirty_network.csv"
    dirty_network.to_csv(filepath, index=False)

    clean_df = load_network_log(filepath)

    assert len(clean_df) == 1
    assert clean_df.iloc[0]["txid"] == "a" * 64