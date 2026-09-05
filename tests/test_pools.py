import pytest

from src.generate.pools import assemble_pools, validate_pools


def test_validate_pools():
    pools = assemble_pools()
    assert validate_pools(pools) is True


def test_validate_pools_rejects_scenario_id_mismatch():
    broken_pools = assemble_pools()
    _, broken_ransomware, _, _ = broken_pools

    victims, _, _ = broken_ransomware[0]
    victims[0].scenario_id = "R999"

    with pytest.raises(
        ValueError,
        match="scenario_id mismatch",
    ):
        validate_pools(broken_pools)