import os
import random
import pandas as pd
from pathlib import Path
from datetime import timedelta
from .config import (
    RANDOM_SEED,
    N_LEGIT_TRANSACTIONS,
    START_DATE,
    TIME_RANGE_DAYS,
    RANSOMWARE_CAMPAIGN_DAYS_RANGE,
    RANSOM_DEMAND_AMOUNTS,
    PEELING_STARTING_LUMP_SUM_RANGE,
    MIXING_ROUNDS_PER_CLUSTER_RANGE,
    MIXING_DENOMINATIONS,
)
from .pools import assemble_pools, validate_pools, build_mixing_cluster, _flatten


PROJECT_ROOT = Path(__file__).resolve().parents[2]
def generate_txid():
    #Generate a synthetic 64-character hex transaction ID.
    tx_id = random.getrandbits(256)
    return f"{tx_id:064x}"


def generate_legit_transactions(legit_pool):
    
    #Generate synthetic legit Bitcoin transactions, returns a list of transaction dicts.
    transactions = []

    for _ in range(N_LEGIT_TRANSACTIONS):
        source = random.choice(legit_pool)

        output_count = random.randint(1, 3)
        outputs = random.sample(legit_pool, output_count)
        while source in outputs:
            outputs = random.sample(legit_pool, output_count)
        output_amounts = [random.lognormvariate(0, 1) for _ in range(output_count)]

        txid = generate_txid()
        fee = random.expovariate(10000)
        input_amounts = [sum(output_amounts) + fee]

        random_offset_seconds = random.uniform(0, TIME_RANGE_DAYS * 86400)
        timestamp = START_DATE + timedelta(seconds=random_offset_seconds)

        transaction = {
            "txid": txid,
            "timestamp": timestamp,
            "input_addresses": [source.address],
            "output_addresses": [w.address for w in outputs],
            "input_amounts": input_amounts,
            "output_amounts": output_amounts,
            "fee": fee,
            "script_type": source.script_type
        }

        transactions.append(transaction)

    return transactions

def generate_sweep_transaction(collector, sweep_destination, total_collected, last_timestamp):
    
    #Generate the collector's final sweep to a fresh downstream wallet,shortly after latest victim payment.
    sweep_fee = min(random.expovariate(10000), total_collected * 0.05)
    sweep_timestamp = last_timestamp + timedelta(hours=2)

    return {
        "txid": generate_txid(),
        "timestamp": sweep_timestamp,
        "input_addresses": [collector.address],
        "output_addresses": [sweep_destination.address],
        "input_amounts": [total_collected],
        "output_amounts": [total_collected - sweep_fee],
        "fee": sweep_fee,
        "script_type": collector.script_type
    }


def generate_ransomware_transactions(victims, collector, sweep_destination):
     
    # Generate ransomware payment transactions: each victim pays the collector, collector sweeps funds to a fresh downstream wallet shortly
    if not victims:
        raise ValueError("generate_ransomware_transactions() requires at least one victim")

    transactions = []
    total_collected = 0
    latest_timestamp = None

    campaign_days = random.randint(*RANSOMWARE_CAMPAIGN_DAYS_RANGE)

    for victim in victims:
        amt_to_pay = random.choice(RANSOM_DEMAND_AMOUNTS)

        jitter_multiplier = random.uniform(0.9, 1.1)
        jittered_amount = amt_to_pay * jitter_multiplier

        # fee capped at 5% of this payment, to guarantee a positive output
        fee = min(random.expovariate(10000), jittered_amount * 0.05)
        total_collected += jittered_amount - fee

        random_offset_seconds = random.uniform(0, campaign_days * 86400)
        timestamp = START_DATE + timedelta(seconds=random_offset_seconds)

        if latest_timestamp is None or timestamp > latest_timestamp:
            latest_timestamp = timestamp

        transaction = {
            "txid": generate_txid(),
            "timestamp": timestamp,
            "input_addresses": [victim.address],
            "output_addresses": [collector.address],
            "input_amounts": [jittered_amount],
            "output_amounts": [jittered_amount - fee],
            "fee": fee,
            "script_type": victim.script_type
        }

        transactions.append(transaction)

    sweep_transaction = generate_sweep_transaction(
        collector,
        sweep_destination,
        total_collected,
        latest_timestamp
    )

    transactions.append(sweep_transaction)

    return transactions


def generate_peeling_transactions(starting_source, hops):

    # Generate a peeling chain where each hop peels an amount and passes the remainder to the next change wallet; the chain ends early if a hop cannot produce positive amounts.
    transactions = []

    current_input_wallet = starting_source
    remaining_balance = random.uniform(*PEELING_STARTING_LUMP_SUM_RANGE)
    current_timestamp = START_DATE + timedelta(
        seconds=random.uniform(0, TIME_RANGE_DAYS * 86400)
    )

    for change_wallet, peeled_destination in hops:
        fee = min(random.expovariate(10000), remaining_balance * 0.05)
        spendable = remaining_balance - fee

        peel_fraction = random.uniform(0.02, 0.10)
        peeled_amount = spendable * peel_fraction
        change_amount = spendable - peeled_amount

        if change_amount <= 0 or peeled_amount <= 0:
            break

        current_timestamp += timedelta(minutes=random.uniform(2, 30))

        transaction = {
            "txid": generate_txid(),
            "timestamp": current_timestamp,
            "input_addresses": [current_input_wallet.address],
            "output_addresses": [peeled_destination.address, change_wallet.address],
            "input_amounts": [remaining_balance],
            "output_amounts": [peeled_amount, change_amount],
            "fee": fee,
            "script_type": current_input_wallet.script_type
        }
        transactions.append(transaction)

        current_input_wallet = change_wallet
        remaining_balance = change_amount

    return transactions


def generate_mixing_transactions(scenario_id, initial_participants, mixing_pool_builder):

    # Generate mixing-cluster transactions across multiple rounds, chaining each round's outputs into the next and retaining only wallets used in transactions.
    transactions = []
    extra_wallets = []

    rounds = random.randint(*MIXING_ROUNDS_PER_CLUSTER_RANGE)
    denomination = random.choice(MIXING_DENOMINATIONS)

    current_inputs = initial_participants
    current_amounts = [denomination] * len(current_inputs)
    current_timestamp = START_DATE + timedelta(
        seconds=random.uniform(0, TIME_RANGE_DAYS * 86400)
    )

    for _ in range(rounds):
        round_outputs_raw = mixing_pool_builder(scenario_id)
        pair_count = min(len(current_inputs), len(round_outputs_raw))

        if pair_count == 0:
            break  # no wallets to pair this round, stop this cluster's mixing here

        round_outputs = round_outputs_raw[:pair_count]
        extra_wallets.extend(round_outputs)

        current_timestamp += timedelta(hours=random.uniform(1, 12))

        next_amounts = []
        for i in range(pair_count):
            input_wallet = current_inputs[i]
            output_wallet = round_outputs[i]
            input_amount = current_amounts[i]

            fee = min(random.expovariate(10000), input_amount * 0.05)
            output_amount = input_amount - fee

            tx_timestamp = current_timestamp + timedelta(
                seconds=random.uniform(-300, 300)
            )

            transaction = {
                "txid": generate_txid(),
                "timestamp": tx_timestamp,
                "input_addresses": [input_wallet.address],
                "output_addresses": [output_wallet.address],
                "input_amounts": [input_amount],
                "output_amounts": [output_amount],
                "fee": fee,
                "script_type": input_wallet.script_type
            }
            transactions.append(transaction)
            next_amounts.append(output_amount)

        current_inputs = round_outputs
        current_amounts = next_amounts

    return transactions, extra_wallets


def build_ledger():
    # Builds all wallet pools and transaction types, then combines them into one ledger in timestamp order. It uses the same random seed each time so the generated ledger is reproducible.
    random.seed(RANDOM_SEED)

    pools = assemble_pools()
    validate_pools(pools)

    legit_pool, ransomware_clusters, peeling_chains, mixing_clusters = pools

    all_rows = []
    extra_wallets = []

    all_rows.extend(generate_legit_transactions(legit_pool))

    for victims, collector, sweep_destination in ransomware_clusters:
        all_rows.extend(generate_ransomware_transactions(victims, collector, sweep_destination))

    for starting_source, hops in peeling_chains:
        all_rows.extend(generate_peeling_transactions(starting_source, hops))

    for participants in mixing_clusters:
        scenario_id = participants[0].scenario_id
        mixing_txns, mixing_extra_wallets = generate_mixing_transactions(
            scenario_id, participants, build_mixing_cluster
        )
        all_rows.extend(mixing_txns)
        extra_wallets.extend(mixing_extra_wallets)

    ledger_df = pd.DataFrame(all_rows)
    ledger_df = ledger_df.sort_values("timestamp").reset_index(drop=True)

    return ledger_df, pools, extra_wallets


def validate_ledger(ledger_df, pools, extra_wallets=None):
     
    # Ledger-level validation gate. Raises loudly, never silently repairs.
    required_columns = {
        "txid", "timestamp", "input_addresses", "output_addresses",
        "input_amounts", "output_amounts", "fee", "script_type"
    }
    missing = required_columns - set(ledger_df.columns)
    if missing:
        raise ValueError(f"Ledger missing required columns: {missing}")

    if ledger_df["txid"].duplicated().any():
        raise ValueError("Duplicate txid(s) found in ledger")

    known_wallets = _flatten(pools)
    known_addresses = {w.address for w in known_wallets}
    if extra_wallets:
        known_addresses |= {w.address for w in extra_wallets}

    for idx, row in ledger_df.iterrows():
        for addr in row["input_addresses"] + row["output_addresses"]:
            if addr not in known_addresses:
                raise ValueError(f"Row {idx}: address {addr} not found in wallet pools")

        if any(a <= 0 for a in row["input_amounts"]):
            raise ValueError(f"Row {idx}: non-positive input amount")
        if any(a <= 0 for a in row["output_amounts"]):
            raise ValueError(f"Row {idx}: non-positive output amount")
        if row["fee"] <= 0:
            raise ValueError(f"Row {idx}: non-positive fee")

        window_end = START_DATE + timedelta(days=TIME_RANGE_DAYS + 5)
        if not (START_DATE <= row["timestamp"] <= window_end):
            raise ValueError(f"Row {idx}: timestamp {row['timestamp']} outside window")

        total_in = sum(row["input_amounts"])
        total_out = sum(row["output_amounts"])
        if total_out >= total_in:
            raise ValueError(f"Row {idx}: output total >= input total")
        if abs(total_in - (total_out + row["fee"])) > 1e-8:
            raise ValueError(f"Row {idx}: input != output + fee")

    return True


def write_ledger_outputs(ledger_df, pools, extra_wallets=None):
    
    # Write the ledger CSV and a separate ground-truth wallet lookup CSV, kept apart so the ground truth is never joinable into model input.
    clean_dir = PROJECT_ROOT / "data" / "clean"
    ground_truth_dir = PROJECT_ROOT / "data" / "ground_truth"

    clean_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    ledger_df.to_csv(clean_dir / "ledger.csv", index=False)

    all_wallets = _flatten(pools)
    if extra_wallets:
        all_wallets = all_wallets + extra_wallets

    
    wallet_metadata_df = pd.DataFrame(
      [
        {
            "address": w.address,
            "script_type": w.script_type,
        }
        for w in all_wallets
      ]
    )

    wallet_metadata_df.to_csv(
       clean_dir / "wallet_metadata.csv",
       index=False,
    )

    ground_truth_df = pd.DataFrame([
        {
            "wallet_id": w.wallet_id,
            "address": w.address,
            "scenario": w.scenario,
            "scenario_id": w.scenario_id,
            "role": w.role,
        }
        for w in all_wallets
    ])
    ground_truth_df.to_csv(ground_truth_dir / "wallet_ground_truth.csv", index=False)
