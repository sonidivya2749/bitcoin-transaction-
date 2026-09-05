import ipaddress
import random


def random_public_ip():
    while True:
        ip = ipaddress.IPv4Address(
            random.randint(0x01000000, 0xDFFFFFFF)
        )

        if ip.is_global:
            return str(ip)


def assign_ip_profile(category):

    if category == "legit":
        home_ips = [
            random_public_ip()
            for _ in range(random.choice([1, 1, 1, 2]))
        ]
        return lambda: random.choice(home_ips)

    elif category == "naive_illicit":
        home_ip = random_public_ip()
        return lambda: home_ip

    elif category == "evasive_illicit":
        return random_public_ip

    else:
        raise ValueError( f"INVALID IP PROFILE CATEGORY: {category}" )