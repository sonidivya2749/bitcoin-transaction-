import random
from bip_utils import (
    Bip44, Bip44Coins,
    Bip49, Bip49Coins,
    Bip84, Bip84Coins,
    Bip86, Bip86Coins,
)

SCRIPT_WEIGHTS = {
    "P2PKH": 0.35,
    "P2SH": 0.20,
    "P2WPKH": 0.35,
    "P2TR": 0.10,
}

_DERIVERS = {
    "P2PKH": (Bip44, Bip44Coins.BITCOIN),
    "P2SH": (Bip49, Bip49Coins.BITCOIN),
    "P2WPKH": (Bip84, Bip84Coins.BITCOIN),
    "P2TR": (Bip86, Bip86Coins.BITCOIN),
}


def verify_scripttype(script_type):
    if script_type in SCRIPT_WEIGHTS:
        return script_type
    raise ValueError(f"INVALID SCRIPT TYPE: {script_type}")


def generate_wallet(script_type=None):
    # Random 32-byte private key, direct BIP44/49/84/86 derivation.

    if script_type is not None:
        script_type = verify_scripttype(script_type)
    else:
        script_type = random.choices(
            list(SCRIPT_WEIGHTS.keys()),
            weights=SCRIPT_WEIGHTS.values(),
        )[0]

    priv_key_bytes = random.randbytes(32)
    bip_cls, coin = _DERIVERS[script_type]

    wallet = bip_cls.FromPrivateKey(priv_key_bytes, coin)
    address = (
        wallet
        .DeriveDefaultPath()
        .PublicKey()
        .ToAddress()
    )

    return address, script_type