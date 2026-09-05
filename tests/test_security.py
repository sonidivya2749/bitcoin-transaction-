import pytest
from defusedxml import ElementTree as ET

from src.ingestion.security import (
    MAX_FILE_SIZE_MB,
    MAX_ROWS,
    validate_file_security,
    validate_row_limit,
    validate_depth,
    validate_json_depth,
    validate_xml_depth,
)


def test_valid_csv_file(tmp_path):
    # Verify a valid CSV file is accepted.
    filepath = tmp_path / "data.csv"
    filepath.write_text("txid,fee\nabc,0.1\n", encoding="utf-8",)
    assert validate_file_security(filepath) == filepath.resolve()


def test_unsupported_file_format(tmp_path):
    # Verify unsupported file formats are rejected.
    filepath = tmp_path / "data.txt"
    filepath.write_text("test", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Unsupported file format",
    ):
        validate_file_security(filepath)


def test_missing_file(tmp_path):
    # Verify missing files are rejected.
    filepath = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Input file not found",
    ):
        validate_file_security(filepath)


def test_empty_file(tmp_path):
    # Verify empty files are rejected.
    filepath = tmp_path / "empty.csv"
    filepath.touch()

    with pytest.raises(
        ValueError,
        match="Input file is empty",
    ):
        validate_file_security(filepath)


def test_directory_rejected(tmp_path):
    # Verify directories are rejected.
    filepath = tmp_path / "data.csv"
    filepath.mkdir()

    with pytest.raises(
        ValueError,
        match="Input path is not a file",
    ):
        validate_file_security(filepath)


def test_file_size_limit(tmp_path):
    # Verify oversized files are rejected.
    filepath = tmp_path / "large.csv"
    filepath.write_bytes(b"x" * (MAX_FILE_SIZE_MB * 1024 * 1024 + 1))

    with pytest.raises(
        ValueError,
        match="File exceeds maximum size",
    ):
        validate_file_security(filepath)


def test_row_limit():
    # Verify the maximum row limit is enforced.
    assert validate_row_limit(MAX_ROWS, "test.csv",) is True

    with pytest.raises(
        ValueError,
        match="rows exceed the maximum allowed",
    ):
        validate_row_limit(
            MAX_ROWS + 1,
            "test.csv",
        )


def test_validate_depth():
    # Verify the generic depth check works.
    assert validate_depth(10, 10,"JSON",) is True

    with pytest.raises(
        ValueError,
        match="nesting depth exceeds",
    ):
        validate_depth(
            11,
            10,
            "JSON",
        )


def test_json_depth():
    # Verify deeply nested JSON is rejected.
    valid_data = {
        "a": {
            "b": 1,
        },
    }

    assert validate_json_depth(
        valid_data,
        3,
    ) is True

    invalid_data = {
        "a": {
            "b": {
                "c": 1,
            },
        },
    }

    with pytest.raises(
        ValueError,
        match="JSON",
    ):
        validate_json_depth(
            invalid_data,
            3,
        )

def test_xml_depth():
    # Verify deeply nested XML is rejected.
    valid_xml = ET.fromstring("<root><child><leaf /></child></root>")

    assert validate_xml_depth(valid_xml, 3,) is True
    invalid_xml = ET.fromstring("<root><child><leaf><deep /></leaf></child></root>")

    with pytest.raises(
        ValueError,
        match="XML",
    ):
        validate_xml_depth(
            invalid_xml,
            3,
        )


def test_json_content_mismatch(tmp_path):
    # Verify a JSON extension containing PDF data is rejected.
    filepath = tmp_path / "data.json"
    filepath.write_bytes(b"%PDF-1.7\n")

    with pytest.raises(
        ValueError,
        match="File content does not match .json format",
    ):
        validate_file_security(filepath)


def test_xml_content_mismatch(tmp_path):
    # Verify an XML extension containing PDF data is rejected.
    filepath = tmp_path / "data.xml"
    filepath.write_bytes(b"%PDF-1.7\n")

    with pytest.raises(
        ValueError,
        match="File content does not match .xml format",
    ):
        validate_file_security(filepath)


def test_binary_content_rejected(tmp_path):
    # Verify a CSV extension containing PDF data is rejected.
    filepath = tmp_path / "data.csv"
    filepath.write_bytes(b"%PDF-1.7\n")

    with pytest.raises(
        ValueError,
        match="File content does not match .csv format",
    ):
        validate_file_security(filepath)