import json
import hashlib
import os
import secrets

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")


def _load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash_password(password, salt):
    return hashlib.sha256((salt + password).encode()).hexdigest()


def signup(username, password, full_name=""):
    users = _load_users()
    username = username.strip().lower()
    if not username or not password:
        return False, "Username and password cannot be empty."
    if username in users:
        return False, "That username is already taken."
    salt = secrets.token_hex(16)
    users[username] = {
        "salt": salt,
        "password_hash": _hash_password(password, salt),
        "full_name": full_name or username,
    }
    _save_users(users)
    return True, "Account created. You can log in now."


def login(username, password):
    users = _load_users()
    username = username.strip().lower()
    record = users.get(username)
    if not record:
        return False, "No account with that username.", None
    check = _hash_password(password, record["salt"])
    if check != record["password_hash"]:
        return False, "Incorrect password.", None
    return True, "Welcome back.", record.get("full_name", username)
