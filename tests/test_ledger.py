import pandas as pd
import pytest

from src.generate.config import START_DATE, TIME_RANGE_DAYS
from src.generate.pools import (
    assemble_pools,
    validate_pools,
    build_mixing_cluster,
)

from src.generate.ledger import (
    generate_txid,
    generate_legit_transactions,
    generate_sweep_transaction,
    generate_ransomware_transactions,
    generate_peeling_transactions,
    generate_mixing_transactions,
    build_ledger,
    validate_ledger,
    write_ledger_outputs,
)

from src.generate.config import (
    START_DATE,
    TIME_RANGE_DAYS,
)

@pytest.fixture(scope="module")
def ledger_data():
    return build_ledger()

def test_validate_ledger_rejects_missing_column(ledger_data):
    # Verify missing required columns are rejected.
    ledger_df, pools, extra_wallets =  ledger_data

    ledger_df = ledger_df.drop(columns=["fee"])

    with pytest.raises(
        ValueError,
        match="Ledger missing required columns",
    ):
        validate_ledger(
            ledger_df,
            pools,
            extra_wallets,
        )


def test_validate_ledger_rejects_duplicate_txid(ledger_data):
    # Verify duplicate TXIDs are rejected.
    ledger_df, pools, extra_wallets = ledger_data
    ledger_df = ledger_df.copy()

    ledger_df.loc[1, "txid"] = ledger_df.loc[0, "txid"]

    with pytest.raises(
        ValueError,
        match="Duplicate txid",
    ):
        validate_ledger(
            ledger_df,
            pools,
            extra_wallets,
        )


def test_validate_ledger_rejects_unknown_wallet(ledger_data):
    # Verify unknown wallet addresses are rejected.
    ledger_df, pools, extra_wallets = ledger_data
    ledger_df = ledger_df.copy()

    ledger_df.at[0, "input_addresses"] = [
        "unknown_wallet_address"
    ]

    with pytest.raises(
        ValueError,
        match="not found in wallet pools",
    ):
        validate_ledger(
            ledger_df,
            pools,
            extra_wallets,
        )


def test_validate_ledger_rejects_non_positive_input(ledger_data):
    # Verify non-positive input amounts are rejected.
    ledger_df, pools, extra_wallets = ledger_data
    ledger_df = ledger_df.copy()

    ledger_df.at[0, "input_amounts"] = [0]

    with pytest.raises(
        ValueError,
        match="non-positive input amount",
    ):
        validate_ledger(
            ledger_df,
            pools,
            extra_wallets,
        )


def test_validate_ledger_rejects_non_positive_output(ledger_data):
    # Verify non-positive output amounts are rejected.
    ledger_df, pools, extra_wallets = ledger_data
    ledger_df = ledger_df.copy()

    ledger_df.at[0, "output_amounts"] = [0]

    with pytest.raises(
        ValueError,
        match="non-positive output amount",
    ):
        validate_ledger(
            ledger_df,
            pools,
            extra_wallets,
        )


def test_validate_ledger_rejects_non_positive_fee(ledger_data):
    # Verify non-positive fees are rejected.
    ledger_df, pools, extra_wallets = ledger_data
    ledger_df = ledger_df.copy()

    ledger_df.at[0, "fee"] = 0

    with pytest.raises(
        ValueError,
        match="non-positive fee",
    ):
        validate_ledger(
            ledger_df,
            pools,
            extra_wallets,
        )


def test_validate_ledger_rejects_timestamp_before_window(ledger_data):
    # Verify timestamps before the allowed window are rejected.
    ledger_df, pools, extra_wallets = ledger_data
    ledger_df = ledger_df.copy()

    ledger_df.at[0, "timestamp"] = (
        START_DATE - pd.Timedelta(seconds=1)
    )

    with pytest.raises(
        ValueError,
        match="outside window",
    ):
        validate_ledger(
            ledger_df,
            pools,
            extra_wallets,
        )


def test_validate_ledger_rejects_timestamp_after_window(ledger_data):
    # Verify timestamps after the allowed window are rejected.
    ledger_df, pools, extra_wallets = ledger_data
    ledger_df = ledger_df.copy()

    ledger_df.at[0, "timestamp"] = (
        START_DATE
        + pd.Timedelta(days=TIME_RANGE_DAYS + 6)
    )

    with pytest.raises(
        ValueError,
        match="outside window",
    ):
        validate_ledger(
            ledger_df,
            pools,
            extra_wallets,
        )


def test_validate_ledger_rejects_output_greater_than_input(ledger_data):
    # Verify output total cannot exceed input total.
    ledger_df, pools, extra_wallets = ledger_data
    ledger_df = ledger_df.copy()

    input_total = sum(
        ledger_df.at[0, "input_amounts"]
    )

    ledger_df.at[0, "output_amounts"] = [
        input_total
    ]

    with pytest.raises(
        ValueError,
        match="output total >= input total",
    ):
        validate_ledger(
            ledger_df,
            pools,
            extra_wallets,
        )


def test_validate_ledger_rejects_balance_mismatch(ledger_data):
    # Verify input must equal output plus fee.
    ledger_df, pools, extra_wallets = ledger_data
    ledger_df = ledger_df.copy()

    output_total = sum(
        ledger_df.at[0, "output_amounts"]
    )
    fee = ledger_df.at[0, "fee"]

    ledger_df.at[0, "input_amounts"] = [
        output_total + fee + 1
    ]

    with pytest.raises(
        ValueError,
        match=r"input != output \+ fee",
    ):
        validate_ledger(
            ledger_df,
            pools,
            extra_wallets,
        )


def test_generate_peeling_transactions_empty_hops(pools=None):
    # Verify an empty peeling chain returns no transactions.
    if pools is None:
        pools = assemble_pools()
        validate_pools(pools)

    starting_source, _ = pools[2][0]

    transactions = generate_peeling_transactions(
        starting_source,
        [],
    )

    assert transactions == []


def test_generate_ransomware_transactions_rejects_empty_victims():
    # Verify ransomware generation requires at least one victim.
    pools = assemble_pools()
    validate_pools(pools)

    _, collector, sweep_destination = pools[1][0]

    with pytest.raises(
        ValueError,
        match="requires at least one victim",
    ):
        generate_ransomware_transactions(
            [],
            collector,
            sweep_destination,
        )


def test_generate_mixing_transactions_empty_builder():
    # Verify mixing stops when no output wallets are available.
    pools = assemble_pools()
    validate_pools(pools)

    participants = pools[3][0]
    scenario_id = participants[0].scenario_id

    def empty_builder(_scenario_id):
        return []

    transactions, extra_wallets = (
        generate_mixing_transactions(
            scenario_id,
            participants,
            empty_builder,
        )
    )

    assert transactions == []
    assert extra_wallets == []