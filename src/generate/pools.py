import random
from dataclasses import dataclass
from typing import Callable, Optional

from .wallets import generate_wallet
from .ip_profiles import assign_ip_profile
from .config import (
    RANDOM_SEED,
    N_LEGIT_WALLETS,
    N_RANSOMWARE_CAMPAIGNS,
    RANSOMWARE_VICTIMS_RANGE,
    N_PEELING_CHAINS,
    PEELING_CHAIN_LENGTH_RANGE,
    N_MIXING_CLUSTERS,
    MIXING_INPUTS_PER_ROUND_RANGE,)

@dataclass
class Wallet:
    wallet_id: str
    address: str
    script_type: str
    ip_fn: Callable[[], str]
    ip_behavior: str
    scenario: str
    scenario_id: Optional[str]
    role: str
    
_wallet_counter = 0


def _next_wallet_id():
    # generates a stable internal id, independent of the bitcoin address
    global _wallet_counter
    _wallet_counter += 1
    return f"W{_wallet_counter:05d}"

# valid (scenario, role) combinations, anything else is a construction bug
_VALID_SCENARIO_ROLES = {
    ("legit", "normal"),
    ("ransomware", "victim"),
    ("ransomware", "collector"),
    ("ransomware", "sweep_destination"),
    ("peeling", "change"),
    ("peeling", "peeled_destination"),
    ("mixing", "participant"),
    ("peeling", "starting_source")
}

_VALID_IP_BEHAVIORS = {"legit", "naive_illicit", "evasive_illicit"}

# which ip_behavior values are permitted for a given scenario
_SCENARIO_ALLOWED_IP_BEHAVIORS = {
    "legit": {"legit"},
    "ransomware": {"legit", "evasive_illicit"},
    "peeling": {"naive_illicit", "evasive_illicit"},
    "mixing": {"evasive_illicit"},}


def _make_wallet(scenario, scenario_id, role, ip_category):
    address, script_type = generate_wallet()
    ip_fn = assign_ip_profile(ip_category)

    return Wallet(
        wallet_id=_next_wallet_id(),
        address=address,
        script_type=script_type,
        ip_fn=ip_fn,
        ip_behavior=ip_category,
        scenario=scenario,
        scenario_id=scenario_id,
        role=role,
    )


def build_legit_pool():
    # builds the full legit wallet population, no scenario_id attached
    return [
        _make_wallet("legit", None, "normal", "legit")
        for _ in range(N_LEGIT_WALLETS)
    ]

def build_peeling_chain(scenario_id):
    chain_length = random.randint(*PEELING_CHAIN_LENGTH_RANGE)

    starting_source = _make_wallet(
        "peeling",
        scenario_id,
        "starting_source",
        "evasive_illicit"
    )

    hops = []
    for _ in range(chain_length):
        change_category = random.choice(
            ["naive_illicit", "evasive_illicit"]
        )

        change_wallet = _make_wallet("peeling", scenario_id,"change", change_category)
        dest_category = random.choice(["naive_illicit", "evasive_illicit"])
        peeled_destination = _make_wallet("peeling", scenario_id,"peeled_destination",dest_category) 
        hops.append((change_wallet, peeled_destination))

    return starting_source, hops

def build_ransomware_cluster(scenario_id):
    # victims are innocent bystanders, collector is the illicit actor, sweep_destination is where the collector launders funds onward
    victim_count = random.randint(*RANSOMWARE_VICTIMS_RANGE)

    victims = [_make_wallet("ransomware", scenario_id, "victim", "legit") for _ in range(victim_count)]

    collector = _make_wallet("ransomware", scenario_id, "collector", "evasive_illicit")
    sweep_destination = _make_wallet("ransomware", scenario_id, "sweep_destination", "evasive_illicit")

    return victims, collector, sweep_destination


def build_mixing_cluster(scenario_id):
    # builds one round's worth of participants only
    participant_count = random.randint(*MIXING_INPUTS_PER_ROUND_RANGE)

    return [_make_wallet("mixing", scenario_id, "participant", "evasive_illicit") for _ in range(participant_count)]

def assemble_pools():
    # seed first, before any other statement, for full determinism
    random.seed(RANDOM_SEED)

    global _wallet_counter
    _wallet_counter = 0

    legit_pool = build_legit_pool()

    ransomware_clusters = []
    for i in range(N_RANSOMWARE_CAMPAIGNS):
        scenario_id = f"R{i + 1:03d}"
        victims, collector, sweep_destination = build_ransomware_cluster(scenario_id)
        ransomware_clusters.append((victims, collector, sweep_destination))

    peeling_chains = []
    for i in range(N_PEELING_CHAINS):
        scenario_id = f"P{i + 1:03d}"
        starting_source, hops = build_peeling_chain(scenario_id)
        peeling_chains.append((starting_source, hops))

    mixing_clusters = []
    for i in range(N_MIXING_CLUSTERS):
        scenario_id = f"M{i + 1:03d}"
        participants = build_mixing_cluster(scenario_id)
        mixing_clusters.append(participants)

    pools = (legit_pool, ransomware_clusters, peeling_chains, mixing_clusters)

    _check_population_budget(pools)

    return pools


def _check_population_budget(pools):
    # legit count must match exactly, illicit clusters must fall within range
    legit_pool, ransomware_clusters, peeling_chains, mixing_clusters = pools

    if len(legit_pool) != N_LEGIT_WALLETS:
        raise AssertionError(
            f"Legit pool size mismatch: expected exactly {N_LEGIT_WALLETS}, "
            f"got {len(legit_pool)}"
        )

    lo, hi = RANSOMWARE_VICTIMS_RANGE
    for victims, _collector, _sweep_destination in ransomware_clusters:
        if not (lo <= len(victims) <= hi):
            raise AssertionError(
                f"Ransomware cluster victim count {len(victims)} outside "
                f"configured range {RANSOMWARE_VICTIMS_RANGE}"
            )

    lo, hi = PEELING_CHAIN_LENGTH_RANGE
    for starting_source, hops in peeling_chains:
        if not (lo <= len(hops) <= hi):
            raise AssertionError(
                f"Peeling chain length {len(hops)} outside configured "
                f"range {PEELING_CHAIN_LENGTH_RANGE}"
            )

    lo, hi = MIXING_INPUTS_PER_ROUND_RANGE
    for participants in mixing_clusters:
        if not (lo <= len(participants) <= hi):
            raise AssertionError(
                f"Mixing cluster participant count {len(participants)} "
                f"outside configured range {MIXING_INPUTS_PER_ROUND_RANGE}"
            )


def _flatten(pools):
    # Collapses every cluster type into one flat wallet list for validation.
    legit_pool, ransomware_clusters, peeling_chains, mixing_clusters = pools

    all_wallets = list(legit_pool)

    for victims, collector, sweep_destination in ransomware_clusters:
        all_wallets.extend(victims)
        all_wallets.append(collector)
        all_wallets.append(sweep_destination)

    for starting_source, hops in peeling_chains:
        all_wallets.append(starting_source)

        for change_wallet, peeled_destination in hops:
            all_wallets.append(change_wallet)
            all_wallets.append(peeled_destination)

    for participants in mixing_clusters:
        all_wallets.extend(participants)

    return all_wallets


def _validate_uniqueness(all_wallets):
    # no wallet_id or address may repeat across the entire population
    wallet_ids = [w.wallet_id for w in all_wallets]
    if len(wallet_ids) != len(set(wallet_ids)):
        raise ValueError("Duplicate wallet_id found")

    addresses = [w.address for w in all_wallets]
    if len(addresses) != len(set(addresses)):
        raise ValueError("Duplicate wallet address found")


def _validate_scenario_ids(all_wallets):
    # every expected scenario_id must exist and no unexpected one should appear
    scenario_ids = [w.scenario_id for w in all_wallets if w.scenario_id is not None]

    expected_scenario_ids = (
        [f"R{i + 1:03d}" for i in range(N_RANSOMWARE_CAMPAIGNS)]
        + [f"P{i + 1:03d}" for i in range(N_PEELING_CHAINS)]
        + [f"M{i + 1:03d}" for i in range(N_MIXING_CLUSTERS)]
    )

    if set(scenario_ids) != set(expected_scenario_ids):
        raise ValueError(
            f"scenario_id mismatch. Expected {set(expected_scenario_ids)}, "
            f"found {set(scenario_ids)}"
        )


def _validate_label_consistency(all_wallets):
    # catches typos or invalid combinations like an unassigned role
    for w in all_wallets:
        if (w.scenario, w.role) not in _VALID_SCENARIO_ROLES:
            raise ValueError(
                f"Invalid (scenario, role) combination on {w.wallet_id}: "
                f"({w.scenario!r}, {w.role!r})"
            )


def _validate_legit_scenario_id_nullability(all_wallets):
    # legit wallets must never carry a scenario_id, illicit ones always must
    for w in all_wallets:
        if w.scenario == "legit" and w.scenario_id is not None:
            raise ValueError(
                f"Legit wallet {w.wallet_id} has non-null scenario_id "
                f"{w.scenario_id!r}"
            )
        if w.scenario != "legit" and w.scenario_id is None:
            raise ValueError(
                f"Illicit-scenario wallet {w.wallet_id} ({w.scenario}) "
                f"has no scenario_id"
            )


def _validate_referential_integrity(pools):
    # confirms every wallet's scenario_id matches the cluster it actually came from
    legit_pool, ransomware_clusters, peeling_chains, mixing_clusters = pools

    for i, (victims, collector, sweep_destination) in enumerate(ransomware_clusters):
        expected_id = f"R{i + 1:03d}"
        for w in victims + [collector, sweep_destination]:
            if w.scenario_id != expected_id:
                raise ValueError(
                    f"Ransomware cluster {i} contains wallet {w.wallet_id} "
                    f"with mismatched scenario_id {w.scenario_id!r} "
                    f"(expected {expected_id!r})"
                )

    for i, (starting_source, hops) in enumerate(peeling_chains):
        expected_id = f"P{i + 1:03d}"
        if starting_source.scenario_id != expected_id:
            raise ValueError(
                f"Peeling chain {i} starting source has mismatched "
                f"scenario_id {starting_source.scenario_id!r} "
                f"(expected {expected_id!r})"
            )

        for change_wallet, peeled_destination in hops:
            for w in (change_wallet, peeled_destination):
                if w.scenario_id != expected_id:
                    raise ValueError(
                        f"Peeling chain {i} contains wallet {w.wallet_id} "
                        f"with mismatched scenario_id {w.scenario_id!r} "
                        f"(expected {expected_id!r})"
                    )

    for i, participants in enumerate(mixing_clusters):
        expected_id = f"M{i + 1:03d}"
        for w in participants:
            if w.scenario_id != expected_id:
                raise ValueError(
                    f"Mixing cluster {i} contains wallet {w.wallet_id} "
                    f"with mismatched scenario_id {w.scenario_id!r} "
                    f"(expected {expected_id!r})"
                )


def _validate_ip_behavior(all_wallets):
    # confirms ip_behavior is a known category and permitted for its scenario
    for w in all_wallets:
        if w.ip_behavior not in _VALID_IP_BEHAVIORS:
            raise ValueError(
                f"Wallet {w.wallet_id} has invalid ip_behavior "
                f"{w.ip_behavior!r}"
            )

        allowed = _SCENARIO_ALLOWED_IP_BEHAVIORS[w.scenario]
        if w.ip_behavior not in allowed:
            raise ValueError(
                f"Wallet {w.wallet_id}: scenario={w.scenario!r} does not "
                f"permit ip_behavior={w.ip_behavior!r} (allowed: {allowed})"
            )


def _validate_ip_fn_consistency(all_wallets, samples=5):
    # diagnostic only, not called from validate_pools, kept for manual debugging, probabilistic check, can produce false positives on evasive profiles
    for w in all_wallets:
        observed = [w.ip_fn() for _ in range(samples)]

        if w.ip_behavior == "naive_illicit":
            if len(set(observed)) != 1:
                raise ValueError(
                    f"Wallet {w.wallet_id}: ip_behavior='naive_illicit' "
                    f"but ip_fn() produced varying IPs: {set(observed)}"
                )
        elif w.ip_behavior == "evasive_illicit":
            if len(set(observed)) == 1:
                raise ValueError(
                    f"Wallet {w.wallet_id}: ip_behavior='evasive_illicit' "
                    f"but ip_fn() returned the same IP {samples} times in "
                    f"a row, statistically suspicious, check construction"
                )
        # legit is intentionally not checked, a single home ip is valid


def validate_pools(pools):
    # the mandatory validation gate, every check raises loudly on failure
    all_wallets = _flatten(pools)

    _validate_uniqueness(all_wallets)
    _validate_scenario_ids(all_wallets)
    _validate_label_consistency(all_wallets)
    _validate_legit_scenario_id_nullability(all_wallets)
    _validate_referential_integrity(pools)
    _validate_ip_behavior(all_wallets)

    return True


 