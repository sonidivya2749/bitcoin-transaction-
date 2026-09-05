import ipaddress

import pytest

from src.generate.ip_profiles import (
    random_public_ip,
    assign_ip_profile,
)


def test_random_public_ip_returns_valid_public_ipv4():
    # Verify generated IP is a valid public IPv4 address.
    ip = ipaddress.IPv4Address(random_public_ip())

    assert ip.version == 4
    assert ip.is_global


def test_legit_profile_returns_public_ip():
    # Verify legitimate profile returns a public IP.
    get_ip = assign_ip_profile("legit")
    ip = ipaddress.IPv4Address(get_ip())

    assert ip.version == 4
    assert ip.is_global


def test_naive_illicit_profile_reuses_same_ip():
    # Verify naive illicit profile keeps the same IP.
    get_ip = assign_ip_profile("naive_illicit")

    ips = [get_ip() for _ in range(10)]

    assert len(set(ips)) == 1


def test_evasive_illicit_profile_generates_different_ips():
    # Verify evasive illicit profile generates changing IPs.
    get_ip = assign_ip_profile("evasive_illicit")

    ips = [get_ip() for _ in range(10)]

    assert len(set(ips)) > 1


def test_legit_profile_uses_limited_home_ips():
    # Verify legitimate profile uses at most two home IPs.
    get_ip = assign_ip_profile("legit")

    ips = [get_ip() for _ in range(20)]

    assert 1 <= len(set(ips)) <= 2


def test_invalid_ip_profile_category():
    # Verify unsupported profile categories raise an error.
    with pytest.raises(
        ValueError,
        match="INVALID IP PROFILE CATEGORY",
    ):
        assign_ip_profile("invalid")


def test_all_profile_outputs_are_valid_public_ips():
    # Verify every supported profile returns public IPv4 addresses.
    for category in ["legit", "naive_illicit", "evasive_illicit"]:
        get_ip = assign_ip_profile(category)

        for _ in range(5):
            ip = ipaddress.IPv4Address(get_ip())

            assert ip.version == 4
            assert ip.is_global