import ast
import ipaddress
import math
import time
import warnings
import re
from functools import lru_cache
from numbers import Real

import pandas as pd
from bip_utils import CoinsConf, P2PKHAddrDecoder, P2SHAddrDecoder, P2TRAddrDecoder, P2WPKHAddrDecoder

from .schema import LEDGER_COLUMNS, NETWORK_LOG_COLUMNS, check_columns,WALLET_COLUMNS


TXID_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
LEDGER_LIST_COLUMNS = ("input_addresses", "output_addresses", "input_amounts", "output_amounts")
SUPPORTED_SCRIPT_TYPES = frozenset({"P2PKH", "P2SH", "P2WPKH", "P2TR"})

BITCOIN_MAINNET_P2PKH_VER = CoinsConf.BitcoinMainNet.m_params["p2pkh_net_ver"]
BITCOIN_MAINNET_P2SH_VER = CoinsConf.BitcoinMainNet.m_params["p2sh_net_ver"]
BITCOIN_MAINNET_SEGWIT_HRP = CoinsConf.BitcoinMainNet.m_params["p2wpkh_hrp"]
BITCOIN_MAINNET_TAPROOT_HRP = CoinsConf.BitcoinMainNet.m_params["p2tr_hrp"]

ISO_COUNTRY_PATTERN = re.compile(r"^[A-Z]{2}$")
ASN_PATTERN = re.compile(r"^AS[0-9]+$")


def _parse_list(value, column, index):
    # Parse list values serialized in CSV-compatible form.
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"{column}: invalid list at row {index}: {value!r}") from exc
        if isinstance(parsed, list):
            return parsed

    raise ValueError(f"{column}: expected a list at row {index}, got {type(value).__name__}")


def _validate_txids(df, name):
    # Validate TXID format.
    valid = df["txid"].apply(lambda value: bool(TXID_PATTERN.fullmatch(str(value).strip())))
    if not valid.all():
        rows = df.index[~valid].tolist()[:5]
        raise ValueError(f"{name}: malformed txid at row(s): {rows}")


def _validate_timestamps(df, name):
    # Validate and normalize timestamps.
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    except Exception as exc:
        raise ValueError(f"{name}: invalid timestamp: {exc}") from exc
    return df


def _validate_no_nulls(df, columns, name):
    # Reject nulls in critical fields.
    for column in columns:
        mask = df[column].isna()
        if mask.any():
            rows = df.index[mask].tolist()[:5]
            raise ValueError(f"{name}: null value in '{column}' at row(s): {rows}")


def _validate_positive_number(value, column, index):
    # Validate finite positive numeric values.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{column}: non-numeric value at row {index}: {value!r}")

    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{column}: invalid value at row {index}: {value!r}")


@lru_cache(maxsize=None)
def _is_global_ip(value):
    # Pure check, cached by value: IPs repeat heavily across rows
    # (e.g. legit wallets reuse 1-2 home IPs across many transactions),
    # so caching turns repeat lookups into O(1) instead of re-parsing.
    # Returns None for unparseable input, True/False for global routability.
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError:
        return None


def _validate_ip(value, column, index):
    # Validate globally routable IP addresses.
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{column}: invalid IP at row {index}: {value!r}")

    stripped = value.strip()
    is_global = _is_global_ip(stripped)

    if is_global is None:
        raise ValueError(f"{column}: invalid IP at row {index}: {value!r}")

    if not is_global:
        raise ValueError(f"{column}: IP is not globally routable at row {index}: {value!r}")


@lru_cache(maxsize=None)
def _is_valid_bitcoin_address(address):
    # Pure check, cached by value: the same wallet address appears across
    # many transactions (mixing clusters, peeling chains, reused legit
    # wallets), so caching avoids re-running up to 4 bip_utils decode
    # attempts on an address we've already validated.
    decoders = (
        (P2PKHAddrDecoder, {"net_ver": BITCOIN_MAINNET_P2PKH_VER}),
        (P2SHAddrDecoder, {"net_ver": BITCOIN_MAINNET_P2SH_VER}),
        (P2WPKHAddrDecoder, {"hrp": BITCOIN_MAINNET_SEGWIT_HRP}),
        (P2TRAddrDecoder, {"hrp": BITCOIN_MAINNET_TAPROOT_HRP}),
    )

    for decoder, kwargs in decoders:
        try:
            decoder.DecodeAddr(address, **kwargs)
            return True
        except (ValueError, TypeError):
            continue

    return False


def _validate_bitcoin_address(value, column, index):
    # Validate a Bitcoin mainnet address using bip_utils.
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{column}: invalid Bitcoin address at row {index}: {value!r}")

    address = value.strip()

    if _is_valid_bitcoin_address(address):
        return address

    raise ValueError(f"{column}: invalid Bitcoin mainnet address at row {index}: {address!r}")


def _validate_script_type(value, index):
    # Validate supported Bitcoin script types.
    if not isinstance(value, str):
        raise ValueError(f"script_type: invalid value at row {index}: {value!r}")

    script_type = value.strip()
    if script_type not in SUPPORTED_SCRIPT_TYPES:
        raise ValueError(f"script_type: unsupported value at row {index}: {value!r}")

    return script_type


def _validate_geo_country(value, index):
    # Validate GeoIP country code or the generator's UNKNOWN fallback.
    if not isinstance(value, str):
        raise ValueError(f"network_log: invalid geo_country at row {index}: {value!r}")

    country = value.strip().upper()
    if country != "UNKNOWN" and not ISO_COUNTRY_PATTERN.fullmatch(country):
        raise ValueError(f"network_log: invalid geo_country at row {index}: {value!r}")

    return country


def _validate_asn(value, index):
    # Validate ASN format or the generator's UNKNOWN fallback.
    if not isinstance(value, str):
        raise ValueError(f"network_log: invalid asn at row {index}: {value!r}")

    asn = value.strip().upper()
    if asn != "UNKNOWN" and not ASN_PATTERN.fullmatch(asn):
        raise ValueError(f"network_log: invalid asn at row {index}: {value!r}")

    return asn


def _validate_ledger_row(row, index):
    # Validate and normalize one ledger row.
    values = {column: _parse_list(row[column], column, index) for column in LEDGER_LIST_COLUMNS}
    inputs, outputs = values["input_addresses"], values["output_addresses"]
    input_amounts, output_amounts = values["input_amounts"], values["output_amounts"]

    if not inputs:
        raise ValueError(f"ledger: empty input_addresses at row {index}")
    if not outputs:
        raise ValueError(f"ledger: empty output_addresses at row {index}")
    if len(inputs) != len(input_amounts):
        raise ValueError(f"ledger: input_addresses/input_amounts length mismatch at row {index}")
    if len(outputs) != len(output_amounts):
        raise ValueError(f"ledger: output_addresses/output_amounts length mismatch at row {index}")

    validated_inputs = [_validate_bitcoin_address(v, "input_addresses", index) for v in inputs]
    validated_outputs = [_validate_bitcoin_address(v, "output_addresses", index) for v in outputs]

    validated_input_amounts = []
    for value in input_amounts:
        _validate_positive_number(value, "input_amounts", index)
        validated_input_amounts.append(float(value))

    validated_output_amounts = []
    for value in output_amounts:
        _validate_positive_number(value, "output_amounts", index)
        validated_output_amounts.append(float(value))

    try:
        fee = float(row["fee"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"fee: non-numeric value at row {index}: {row['fee']!r}") from exc

    _validate_positive_number(fee, "fee", index)
    script_type = _validate_script_type(row["script_type"], index)

    total_input, total_output = sum(validated_input_amounts), sum(validated_output_amounts)

    if total_output >= total_input:
        raise ValueError(f"ledger: output total must be less than input total at row {index}")

    if not math.isclose(total_input, total_output + fee, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"ledger: input total does not equal output total + fee at row {index}")

    return (
        validated_inputs,
        validated_outputs,
        validated_input_amounts,
        validated_output_amounts,
        fee,
        script_type,
    )


def validate_ledger(df):
    # Validate and normalize the complete ledger.
    check_columns(df, LEDGER_COLUMNS, "ledger")
    if df.empty:
        raise ValueError("ledger: dataset is empty")

    _validate_no_nulls(df, ["txid", "timestamp", "fee"], "ledger")
    _validate_txids(df, "ledger")

    if df["txid"].duplicated().any():
        raise ValueError("ledger: duplicate txid found")

    df = _validate_timestamps(df, "ledger")

    input_addresses, output_addresses = [], []
    input_amounts_col, output_amounts_col = [], []
    fees, script_types = [], []

    total_rows = len(df)
    loop_start = time.time()

    for position, (index, row) in enumerate(df.iterrows(), start=1):
        inputs, outputs, input_amounts, output_amounts, fee, script_type = _validate_ledger_row(row, index)
        input_addresses.append(inputs)
        output_addresses.append(outputs)
        input_amounts_col.append(input_amounts)
        output_amounts_col.append(output_amounts)
        fees.append(fee)
        script_types.append(script_type)

        if position % 5000 == 0 or position == total_rows:
            print(f"    ledger: validated {position}/{total_rows} rows ({time.time() - loop_start:.1f}s elapsed)")

    # Assign whole columns at once (vectorized) instead of writing one cell
    # at a time via df.at[...] inside the loop above, which is O(n) slow on
    # large datasets (100k+ rows took minutes; this is seconds).
    df["input_addresses"] = pd.Series(input_addresses, index=df.index, dtype=object)
    df["output_addresses"] = pd.Series(output_addresses, index=df.index, dtype=object)
    df["input_amounts"] = pd.Series(input_amounts_col, index=df.index, dtype=object)
    df["output_amounts"] = pd.Series(output_amounts_col, index=df.index, dtype=object)
    df["fee"] = pd.Series(fees, index=df.index, dtype="float64")
    df["script_type"] = pd.Series(script_types, index=df.index, dtype=object)

    return df


def _validate_network_row(row, index):
    # Validate and normalize one network-log row.
    ports = {}

    for column in ("src_port", "dst_port"):
        try:
            value = float(row[column])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"network_log: invalid {column} at row {index}") from exc

        if not math.isfinite(value) or not value.is_integer() or not 1 <= int(value) <= 65535:
            raise ValueError(f"network_log: {column} contains invalid port at row {index}")

        ports[column] = int(value)

    validated_ips = {}
    for column in ("src_ip", "dst_ip"):
        _validate_ip(row[column], column, index)
        validated_ips[column] = str(row[column]).strip()

    geo_country = _validate_geo_country(row["geo_country"], index)
    asn = _validate_asn(row["asn"], index)

    return (
        ports["src_port"],
        ports["dst_port"],
        validated_ips["src_ip"],
        validated_ips["dst_ip"],
        geo_country,
        asn,
    )


def validate_network_log(df):
    # Validate and normalize the complete network log.
    check_columns(df, NETWORK_LOG_COLUMNS, "network_log")
    if df.empty:
        raise ValueError("network_log: dataset is empty")

    _validate_no_nulls(
        df,
        ["txid", "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "geo_country", "asn"],
        "network_log",
    )

    _validate_txids(df, "network_log")
    df = _validate_timestamps(df, "network_log")

    src_ports, dst_ports = [], []
    src_ips, dst_ips = [], []
    geo_countries, asns = [], []

    total_rows = len(df)
    loop_start = time.time()

    for position, (index, row) in enumerate(df.iterrows(), start=1):
        src_port, dst_port, src_ip, dst_ip, geo_country, asn = _validate_network_row(row, index)
        src_ports.append(src_port)
        dst_ports.append(dst_port)
        src_ips.append(src_ip)
        dst_ips.append(dst_ip)
        geo_countries.append(geo_country)
        asns.append(asn)

        if position % 5000 == 0 or position == total_rows:
            print(f"    network_log: validated {position}/{total_rows} rows ({time.time() - loop_start:.1f}s elapsed)")

    # Vectorized column assignment (see validate_ledger for why: per-cell
    # df.at[...] writes inside the loop are O(n) slow on large datasets).
    df["src_port"] = pd.Series(src_ports, index=df.index, dtype="int64")
    df["dst_port"] = pd.Series(dst_ports, index=df.index, dtype="int64")
    df["src_ip"] = pd.Series(src_ips, index=df.index, dtype=object)
    df["dst_ip"] = pd.Series(dst_ips, index=df.index, dtype=object)
    df["geo_country"] = pd.Series(geo_countries, index=df.index, dtype=object)
    df["asn"] = pd.Series(asns, index=df.index, dtype=object)

    return df

def validate_wallets(df):
    # Validate and normalize the complete wallet metadata dataset.
    check_columns(df, WALLET_COLUMNS, "wallet")
    if df.empty:
        raise ValueError("wallet: dataset is empty")

    _validate_no_nulls(df, ["address", "script_type"], "wallet")

    if df["address"].duplicated().any():
        raise ValueError("wallet: duplicate address found")

    for index, row in df.iterrows():
        address = _validate_bitcoin_address(row["address"], "address", index)
        script_type = _validate_script_type(row["script_type"], index)
        df.at[index, "address"] = address
        df.at[index, "script_type"] = script_type

    return df

def validate_cross_file(ledger_df, network_log_df):
    # Validate TXID coverage and relay chronology between datasets.
    ledger_txids = set(ledger_df["txid"])
    network_txids = set(network_log_df["txid"])

    orphan = network_txids - ledger_txids
    if orphan:
        raise ValueError(
            f"network_log: {len(orphan)} txid(s) have no matching ledger transaction, "
            f"e.g. {list(orphan)[:5]}"
        )

    missing = ledger_txids - network_txids
    if missing:
        raise ValueError(
            f"ledger: {len(missing)} transaction(s) have no network observation, "
            f"e.g. {list(missing)[:5]}"
        )

    ledger_times = ledger_df.set_index("txid")["timestamp"]

    for txid, group in network_log_df.groupby("txid", sort=False):
        transaction_time = ledger_times.loc[txid]
        timestamps = group["timestamp"].tolist()

        if timestamps[0] < transaction_time:
            raise ValueError(f"network_log: relay timestamp precedes ledger transaction for txid {txid}")

        for position in range(len(timestamps) - 1):
            if timestamps[position + 1] <= timestamps[position]:
                raise ValueError(f"network_log: relay timestamps are not strictly increasing for txid {txid}")

        src_ips, dst_ips = group["src_ip"].tolist(), group["dst_ip"].tolist()

        for position in range(len(group) - 1):
            if dst_ips[position] != src_ips[position + 1]:
                warnings.warn(
                    f"network_log: broken relay IP continuity for txid {txid} "
                    "(likely caused by a dropped relay hop during row filtering)"
                )
                break
            
    return True


def _build_rejected(df, rows):
    # Build the rejection report.
    if rows:
        return pd.DataFrame(rows)

    return pd.DataFrame(columns=list(df.columns) + ["_row_index", "_rejection_reason"])


def _filter_ledger_row(row, index):
    # Validate one ledger row for filtering.
    if pd.isna(row["txid"]) or not TXID_PATTERN.fullmatch(str(row["txid"]).strip()):
        raise ValueError("invalid txid")

    try:
        timestamp = pd.to_datetime(row["timestamp"], errors="raise")
    except Exception as exc:
        raise ValueError("invalid timestamp") from exc

    inputs, outputs, input_amounts, output_amounts, fee, script_type = _validate_ledger_row(row, index)

    # Cast to object dtype before mutating: a row pulled from an all-string
    # DataFrame (e.g. every field parsed as text from XML) gets a homogeneous
    # Arrow-backed StringDtype, which refuses non-string values like a
    # Timestamp, float, or list assigned below.
    clean = row.astype(object)
    clean["timestamp"] = timestamp
    clean["input_addresses"] = inputs
    clean["output_addresses"] = outputs
    clean["input_amounts"] = input_amounts
    clean["output_amounts"] = output_amounts
    clean["fee"] = fee
    clean["script_type"] = script_type
    return clean

def _filter_network_row(row, index):
    # Validate one network-log row for filtering.
    if pd.isna(row["txid"]) or not TXID_PATTERN.fullmatch(str(row["txid"]).strip()):
        raise ValueError("invalid txid")

    try:
        timestamp = pd.to_datetime(row["timestamp"], errors="raise")
    except Exception as exc:
        raise ValueError("invalid timestamp") from exc

    ports = {}
    for column in ("src_port", "dst_port"):
        try:
            value = float(row[column])
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"invalid {column}")

        if not math.isfinite(value) or not value.is_integer() or not 1 <= int(value) <= 65535:
            raise ValueError(f"invalid {column}")

        ports[column] = int(value)

    _validate_ip(row["src_ip"], "src_ip", index)
    _validate_ip(row["dst_ip"], "dst_ip", index)

    geo_country = _validate_geo_country(row["geo_country"], index)
    asn = _validate_asn(row["asn"], index)

    # See _filter_ledger_row: cast to object dtype before mutating, since an
    # all-string row (e.g. from XML ingestion) gets a StringDtype that
    # rejects non-string values like Timestamp and int.
    clean = row.astype(object)
    clean["timestamp"] = timestamp
    clean["src_port"] = ports["src_port"]
    clean["dst_port"] = ports["dst_port"]
    clean["src_ip"] = str(row["src_ip"]).strip()
    clean["dst_ip"] = str(row["dst_ip"]).strip()
    clean["geo_country"] = geo_country
    clean["asn"] = asn

    return clean

def filter_ledger_rows(df):
    # Keep valid ledger rows and report rejected rows.
    check_columns(df, LEDGER_COLUMNS, "ledger")
    if df.empty:
        raise ValueError("ledger: dataset is empty")

    clean_rows, rejected_rows = [], []

    for index, row in df.iterrows():
        try:
            clean = _filter_ledger_row(row, index)
            clean["_row_index"] = index
            clean_rows.append(clean.to_dict())
        except ValueError as exc:
            rejected = row.copy()
            rejected["_row_index"] = index
            rejected["_rejection_reason"] = str(exc)
            rejected_rows.append(rejected)

    clean_df = pd.DataFrame(
        [{column: row[column] for column in df.columns} for row in clean_rows],
        columns=df.columns,
    )
    rejected_df = _build_rejected(df, rejected_rows)

    if not clean_df.empty:
        duplicate_mask = clean_df["txid"].duplicated(keep="first")

        if duplicate_mask.any():
            duplicate_positions = clean_df.index[duplicate_mask].tolist()
            duplicate_rows = clean_df.loc[duplicate_mask].copy()
            duplicate_rows["_row_index"] = [
                clean_rows[position]["_row_index"] for position in duplicate_positions
            ]
            duplicate_rows["_rejection_reason"] = "duplicate txid"

            rejected_df = pd.concat([rejected_df, duplicate_rows], ignore_index=True)
            clean_df = clean_df.loc[~duplicate_mask].copy()

    return clean_df, rejected_df

def filter_network_rows(df):
    # Keep valid network-log rows and report rejected rows.
    check_columns(df, NETWORK_LOG_COLUMNS, "network_log")
    if df.empty:
        raise ValueError("network_log: dataset is empty")
    
    parsed_timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )
    clean_rows, rejected_rows = [], []

    for position, (index, row) in enumerate(df.iterrows()):
        try:
            if pd.isna(parsed_timestamps.iloc[position]):
               raise ValueError("invalid timestamp")
 
            # Cast to object dtype before mutating: row.copy() alone keeps
            # the original dtype, which crashes on this Timestamp assignment
            # for an all-string row (e.g. from XML ingestion) -- same issue
            # fixed elsewhere in _filter_ledger_row/_filter_network_row.
            row = row.astype(object)
            row["timestamp"] = parsed_timestamps.iloc[position]

            clean = _filter_network_row(row, index)
            clean["_row_index"] = index
            clean_rows.append(clean.to_dict())
        except ValueError as exc:
            rejected = row.copy()
            rejected["_row_index"] = index
            rejected["_rejection_reason"] = str(exc)
            rejected_rows.append(rejected)

    clean_df = pd.DataFrame(
          [{column: row[column] for column in df.columns} for row in clean_rows],
          columns=df.columns,
    )
    rejected_df = _build_rejected(df, rejected_rows)

    return clean_df, rejected_df