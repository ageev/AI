#!/usr/bin/env python3
"""Aggregate a person's food log for today and push their calories widget.

Multi-user: every person has a slot calories_<key>. The primary person
(registry "primary") keeps the historical top-level data
dir; everyone else lives in ~/.hermes/data/food/persons/<key>/. Called
fire-and-forget from foodlog.py after every logged meal (with --user) and
from cron after midnight per person. Prints only non-sensitive status lines
and appends one line per run to ~/hermes-widgets/push.log - the only history
of automatic pushes (foodlog.py runs this hook with stdio devnulled).
"""
import argparse
import json
import os
import subprocess
import sys
import tomllib
from datetime import date, datetime
from pathlib import Path

HW = Path(os.path.expanduser("~/hermes-widgets"))
FOOD_ROOT = Path(os.environ.get("FOODLOG_DIR",
                                os.path.expanduser("~/.hermes/data/food")))
REGISTRY = Path(os.path.expanduser("~/.hermes/data/users.json"))
MEAL_RU = {"breakfast": "Завтрак", "lunch": "Обед", "dinner": "Ужин",
           "snack": "Перекус", "drink": "Напиток"}


def _primary() -> str:
    try:
        return str(json.loads(REGISTRY.read_text()).get("primary") or "alice")
    except Exception:
        return "alice"


def _log_push(user: str, rc, note: str) -> None:
    """Append one status line to push.log. Status only - never payload data."""
    try:
        stamp = datetime.now().replace(microsecond=0).isoformat()
        with open(HW / "push.log", "a", encoding="utf-8") as fh:
            fh.write(f"{stamp} user={user} exit={rc} {note}\n")
    except Exception:
        pass  # logging must never break the push itself


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=None, help="person key (default: primary)")
    args = ap.parse_args()
    user = (args.user or _primary()).strip().lower()

    # symmetric layout: every person (primary included) under persons/<key>
    log = FOOD_ROOT / "persons" / user / "log.jsonl"

    cfg = tomllib.loads((HW / "config.toml").read_text())
    slot = f"calories_{user}"
    slot_cfg = cfg.get("slots", {}).get(slot, {})
    budget = int(slot_cfg.get("budget_kcal", 2000))

    today = date.today().isoformat()
    rows = []
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("date") == today:
                rows.append(e)
    rows.sort(key=lambda e: e.get("ts", ""))

    def num(v):
        return v if isinstance(v, (int, float)) else 0

    payload = {
        "schema": 1,
        "date": today,
        "budget_kcal": budget,
        "eaten_kcal": int(sum(num(e.get("kcal")) for e in rows)),
        "macros": {
            "protein_g": round(sum(num(e.get("protein_g")) for e in rows)),
            "fat_g": round(sum(num(e.get("fat_g")) for e in rows)),
            "carb_g": round(sum(num(e.get("carbs_g")) for e in rows)),
        },
        "meals": [{
            "time": (e.get("ts", "").split("T")[-1][:5]) or "",
            "kcal": int(num(e.get("kcal"))),
            "name": MEAL_RU.get(e.get("meal", ""), "Приём пищи"),
            "item": ", ".join(str(i) for i in (e.get("items") or []))[:60],
        } for e in rows],
    }

    try:
        r = subprocess.run(
            [str(HW / ".venv/bin/python"), str(HW / "widget_push.py"), slot],
            input=json.dumps(payload), text=True, cwd=str(HW),
            capture_output=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        _log_push(user, "timeout", "published=no fcm=none err=widget_push.py exceeded 180s")
        sys.stderr.write("widget push timed out after 180s\n")
        return 1
    out_lines = (r.stdout or "").splitlines()
    for ln in out_lines:
        if ln.startswith(("rendered", "published", "FCM", "local-only")):
            print(ln)
    published = "yes" if any(ln.startswith("published") for ln in out_lines) else "no"
    fcm_line = next((ln for ln in out_lines if ln.startswith("FCM")), "")
    fcm = "sent" if fcm_line.startswith("FCM push sent") else (fcm_line[:40] or "none")
    note = f"published={published} fcm={fcm}"
    if r.returncode:
        err_tail = " ".join((r.stderr or "widget push failed").split())[-120:]
        note += f" err={err_tail}"
        sys.stderr.write((r.stderr or "widget push failed")[-500:] + "\n")
    _log_push(user, r.returncode, note)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
