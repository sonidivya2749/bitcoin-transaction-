import pytest

from src.generate.wallets import (
    SCRIPT_WEIGHTS,
    verify_scripttype,
    generate_wallet,
)


def test_verify_scripttype_accepts_valid_types():
    # Verify all supported script types are accepted.
    for script_type in SCRIPT_WEIGHTS:
        assert verify_scripttype(script_type) == script_type


def test_verify_scripttype_rejects_invalid_type():
    # Verify unsupported script types raise an error.
    with pytest.raises(ValueError, match="INVALID SCRIPT TYPE"):
        verify_scripttype("INVALID")


def test_generate_wallet_with_valid_script_type():
    # Verify wallet generation returns an address and requested script type.
    for script_type in SCRIPT_WEIGHTS:
        address, generated_type = generate_wallet(script_type)

        assert isinstance(address, str)
        assert address
        assert generated_type == script_type


def test_generate_wallet_random_script_type():
    # Verify wallet generation selects one supported script type automatically.
    address, script_type = generate_wallet()

    assert isinstance(address, str)
    assert address
    assert script_type in SCRIPT_WEIGHTS


def test_generate_wallet_addresses_are_unique():
    # Verify separate wallet generations produce different addresses.
    address_1, _ = generate_wallet()
    address_2, _ = generate_wallet()

    assert address_1 != address_2


def test_generate_wallet_rejects_invalid_script_type():
    # Verify invalid script types are rejected before wallet derivation.
    with pytest.raises(ValueError, match="INVALID SCRIPT TYPE"):
        generate_wallet("INVALID")


def test_script_weights_sum_to_one():
    # Verify script selection probabilities form a valid distribution.
    assert sum(SCRIPT_WEIGHTS.values()) == pytest.approx(1.0)