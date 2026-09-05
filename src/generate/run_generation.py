from .ledger import build_ledger, write_ledger_outputs, _flatten
from .network_log import generate_network_log, write_network_log_output


def main():
    ledger_df, pools, extra_wallets = build_ledger()

    write_ledger_outputs(
        ledger_df,
        pools,
        extra_wallets,
    )

    all_wallets = _flatten(pools)

    if extra_wallets:
         all_wallets.extend(extra_wallets)

    network_log_df = generate_network_log(
         ledger_df,
         all_wallets,
    )

    write_network_log_output(network_log_df)


if __name__ == "__main__":
    main()