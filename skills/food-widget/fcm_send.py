"""Send FCM data-messages (HTTP v1) to notify widgets about slot updates.

The Firebase service-account JSON lives ONLY in the vault as a secure note
(`secret note fcm-service-account`); it is loaded in-memory at send time and
never written to disk. Widgets subscribe to topic `hw-<slot>`.
"""
import json
import subprocess

import google.auth.transport.requests
import requests
from google.oauth2 import service_account

VAULT_ITEM = "fcm-service-account"
_SCOPES = ["https://www.googleapis.com/auth/firebase.messaging"]


def _sa_info() -> dict:
    r = subprocess.run(["secret", "note", VAULT_ITEM], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"`secret note {VAULT_ITEM}` failed: "
                           f"{(r.stderr or r.stdout).strip()}")
    return json.loads(r.stdout)


def send_slot_update(slot: str, version: int, updated: str) -> str:
    """Publish a data-only message to topic hw-<slot>; returns the message name."""
    info = _sa_info()
    creds = service_account.Credentials.from_service_account_info(info, scopes=_SCOPES)
    creds.refresh(google.auth.transport.requests.Request())
    payload = {
        "message": {
            "topic": f"hw-{slot}",
            "data": {"slot": slot, "v": str(version), "updated": updated},
            "android": {"priority": "high"},
        }
    }
    resp = requests.post(
        f"https://fcm.googleapis.com/v1/projects/{info['project_id']}/messages:send",
        headers={"Authorization": f"Bearer {creds.token}"},
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("name", "")
