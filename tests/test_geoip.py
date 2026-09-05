import pytest

from src.generate.geoip import (
    ASN_DB_PATH,
    CITY_DB_PATH,
    GeoIPLookup,
)


def test_geoip_databases_exist():
    # Verify both GeoLite2 databases exist.
    assert CITY_DB_PATH.is_file()
    assert ASN_DB_PATH.is_file()


def test_known_ip_lookup():
    # Verify a known public IP resolves to country and ASN.
    with GeoIPLookup() as geoip:
        result = geoip.lookup("8.8.8.8")

    assert result["geo_country"] == "US"
    assert result["asn"] == "AS15169"


def test_unknown_ip_returns_unknown():
    # Verify an IP absent from the database is handled safely.
    with GeoIPLookup() as geoip:
        result = geoip.lookup("192.0.2.1")

    assert result["geo_country"] == "UNKNOWN"
    assert result["asn"] == "UNKNOWN"


def test_invalid_ip_returns_unknown():
    # Verify malformed IP input does not crash the lookup.
    with GeoIPLookup() as geoip:
        result = geoip.lookup("not-an-ip")

    assert result["geo_country"] == "UNKNOWN"
    assert result["asn"] == "UNKNOWN"


def test_missing_city_database(tmp_path):
    # Verify missing City database raises a clear error.
    missing_city = tmp_path / "missing-city.mmdb"

    with pytest.raises(
        FileNotFoundError,
        match="GeoLite2 City database not found",
    ):
        GeoIPLookup(
            city_db_path=missing_city,
            asn_db_path=ASN_DB_PATH,
        )


def test_missing_asn_database(tmp_path):
    # Verify missing ASN database raises a clear error.
    missing_asn = tmp_path / "missing-asn.mmdb"

    with pytest.raises(
        FileNotFoundError,
        match="GeoLite2 ASN database not found",
    ):
        GeoIPLookup(
            city_db_path=CITY_DB_PATH,
            asn_db_path=missing_asn,
        )