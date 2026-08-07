#!/usr/bin/env python3
"""
foodlog.py - deterministic store + analytics for the food-log skill.

The agent's job is only to (1) look at a meal photo and estimate, and
(2) narrate. ALL arithmetic and trend math lives here, in Python, because
the local 27B model is unreliable at summing many rows or computing
week-over-week deltas by itself. The agent calls this tool; this tool does
the counting.

Storage: a JSONL file (one meal per line) as the source of truth, plus
an optional human-readable daily Markdown diary. No database,
no dependencies (stdlib only), trivially backed up and eyeballed.

Data dirs (one per person; override with $FOODLOG_DIR for tests):
    ~/.hermes/data/food/persons/<key>/
        log.jsonl                 # source of truth, one meal per line, APPEND-ONLY
        diary/YYYY-MM-DD.md       # optional human-readable diary
        photos/                   # optional saved photos

All data stays on this host. Nothing here reaches any cloud.

Trust contract for the agent: every command except `report` starts with a
`[foodlog] person=... | now=... | today=...` header. That header is the
authoritative clock and identity for the call - always trust it over any
date/count remembered from the conversation. The log is a shared multi-writer
journal (other chat sessions and household members append at any time), so a
count that differs from what you remember is normal, not corruption.

Subcommands
-----------
  add --json -            Append one meal (JSON on stdin or via --json '<obj>').
                          Prints the stored entry + today's running totals +
                          the trailing 7-day average, so the agent can reply
                          immediately. REFUSES a "ts" that falls on a non-today
                          date unless --backdate is given (guards against stale
                          dates leaking in from conversation context). Prints a
                          NEW DAY banner on the first entry of a new day.
  correct <meal_type>     Find the last entry of the given meal type today,
                          DELETE it, then append the corrected replacement.
                          Use when the user corrects an item ("не томаты а перец").
                          Meal type: breakfast|lunch|dinner|snack|drink.
  today                   Computed summary for today + 7-day comparison.
  day YYYY-MM-DD          Computed summary for a specific day.
  recent [N]              The last N meals (default 6), newest first, for
                          "compare with previous days".
  week                    Last 7 days vs the prior 7 days (trend deltas).
  report [today|week]     Cron-friendly plain-text block. Prints exactly
                          NO_DATA when the period is empty (the cron prompt
                          turns that into [SILENT]).

Meal JSON schema (all fields optional except a description of what was eaten)
    {
      "ts": "2026-07-07T13:20:00",   # optional; filled from clock if absent
      "id": "a3f8b2c1",              # auto-generated UUID (8 chars)
      "meal": "lunch",               # breakfast|lunch|dinner|snack|drink
      "items": ["chicken ~150g", "rice ~200g", "salad"],
      "kcal": 620,
      "protein_g": 45, "carbs_g": 60, "fat_g": 18, "fiber_g": 6,
      "quality": 7,                  # 1-10 nutrient quality (10 = best)
      "flags": ["whole-food", "veg", "protein-rich"],
      "confidence": "medium",        # low|medium|high
      "note": "Balanced, good protein.",
      "photo": "photos/2026-07-07T13-20-00.jpg"   # optional
    }
"""

import argparse
import json
import os
import re
import statistics
import sys
import uuid
from datetime import datetime, timedelta

# --------------------------------------------------------------------------
# per-person data routing (multi-user)
# --------------------------------------------------------------------------
# Each person's food log lives in its own directory so two Telegram users
# never mix meals. Resolution precedence:
#   1. $FOODLOG_DIR  -- absolute override, wins over everything (used by tests).
#   2. --person / $HERMES_PERSON -- a named person key. EVERY person (primary
#      "alice" included) is namespaced under ~/.hermes/data/food/persons/<key>/;
#      an unidentified speaker goes to persons/unknown; no person at all
#      (cron, no session) resolves to the primary person's dir.
# The identity layer sets $HERMES_PERSON per turn from who is talking, so the
# agent normally does not pass --person at all.
FOOD_ROOT = os.path.expanduser("~/.hermes/data/food")
PRIMARY_PERSON = (os.environ.get("FOODLOG_PRIMARY_PERSON") or "").strip().lower()

# Person key explicitly selected via --person / _set_person(); None = current
# speaker. Threaded into timezone resolution so a meal logged FOR someone uses
# THAT person's local clock, not the speaker's.
CURRENT_PERSON = None


def _registry_key():
    """Current person key from the user registry (via $HERMES_SESSION_USER_ID).

    Imported lazily so foodlog stays runnable even if the registry helper is
    absent; any failure degrades to "" (primary user), never a crash.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import person_registry
        return person_registry.resolve_key()
    except Exception:
        return ""


def _primary_key():
    try:
        with open(os.path.expanduser("~/.hermes/data/users.json"),
                  encoding="utf-8") as fh:
            return str(json.load(fh).get("primary") or "alice").strip().lower()
    except Exception:
        return "alice"


def _resolve_data_dir(person=None):
    # persons_always: symmetric layout - EVERY person (primary included)
    # lives under FOOD_ROOT/persons/<key>; unknown speakers under
    # persons/unknown; no person at all (cron) -> the primary's dir.
    override = os.environ.get("FOODLOG_DIR")
    if override:
        return override
    if person is not None:
        key = person
    elif os.environ.get("HERMES_PERSON"):
        key = os.environ["HERMES_PERSON"]
    else:
        # No explicit person: derive it from who is talking to the bot.
        key = _registry_key()
    key = (key or "").strip().lower()
    if not key or key in ("default", "primary"):
        key = _primary_key()
    safe = re.sub(r"[^a-z0-9_-]", "", key) or "unknown"
    return os.path.join(FOOD_ROOT, "persons", safe)


DATA_DIR = _resolve_data_dir()
LOG_PATH = os.path.join(DATA_DIR, "log.jsonl")
DIARY_DIR = os.path.join(DATA_DIR, "diary")


def _set_person(person):
    """Re-point the module storage globals at *person*'s data directory."""
    global DATA_DIR, LOG_PATH, DIARY_DIR, CURRENT_PERSON
    CURRENT_PERSON = person
    DATA_DIR = _resolve_data_dir(person)
    LOG_PATH = os.path.join(DATA_DIR, "log.jsonl")
    DIARY_DIR = os.path.join(DATA_DIR, "diary")


def _person_label():
    """Short person key for output headers ('alice'), or the raw dir when
    storage was overridden with $FOODLOG_DIR (tests)."""
    droot = os.path.abspath(DATA_DIR)
    persons_root = os.path.join(os.path.abspath(FOOD_ROOT), "persons")
    if droot.startswith(persons_root + os.sep):
        return os.path.basename(droot)
    return droot


def _identity_note():
    """Loud line when the person default is about to silently misfile data.

    The dangerous case: a gateway turn (session markers present) whose speaker
    id is missing or unresolvable - reads and writes would quietly land on the
    primary person's journal. Cron and an operator shell (no session markers)
    default to primary BY DESIGN and stay silent; an explicit --person /
    $HERMES_PERSON / $FOODLOG_DIR means the caller chose, so no note either.
    """
    if CURRENT_PERSON is not None or os.environ.get("HERMES_PERSON") \
            or os.environ.get("FOODLOG_DIR"):
        return ""
    in_session = any(os.environ.get(v) for v in (
        "HERMES_SESSION_ID", "HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID"))
    if not in_session:
        return ""
    uid = (os.environ.get("HERMES_SESSION_USER_ID") or "").strip()
    if not uid:
        return (f"WARNING: this gateway session carries NO speaker id - "
                f"operating on the PRIMARY journal ('{_primary_key()}'). If "
                f"this is about someone else's food, re-run with --person "
                f"<key>, and report the lost session identity to the owner.")
    if _registry_key() == "":
        return (f"WARNING: speaker id is set but person_registry failed to "
                f"resolve it - operating on the PRIMARY journal "
                f"('{_primary_key()}'). Check ~/.hermes/scripts/"
                f"person_registry.py and ~/.hermes/data/users.json.")
    if _person_label() == "unknown":
        return ("NOTE: this speaker id is not in ~/.hermes/data/users.json - "
                "entries go to persons/unknown/. Register the person to give "
                "them their own journal.")
    return ""


def _widget_push():
    """Fire the phone widget refresh for the CURRENT person (fire-and-forget).

    Every person has their own widget slot (calories_<key>); the hook gets
    the key explicitly. The primary user's key comes from the registry
    ("primary" field). A widget hiccup must never fail a meal write.
    """
    if os.environ.get("FOODLOG_NO_WIDGET"):
        return
    try:
        droot = os.path.abspath(DATA_DIR)
        persons_root = os.path.join(os.path.abspath(FOOD_ROOT), "persons")
        if droot.startswith(persons_root + os.sep):
            key = os.path.basename(droot)
        else:
            try:
                with open(os.path.expanduser("~/.hermes/data/users.json"),
                          encoding="utf-8") as fh:
                    key = str(json.load(fh).get("primary") or "alice")
            except Exception:
                key = "alice"
        import subprocess
        subprocess.Popen(
            [os.path.expanduser("~/hermes-widgets/hooks/food_widget_push.py"),
             "--user", key],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

NUM_FIELDS = ("kcal", "protein_g", "carbs_g", "fat_g", "fiber_g", "quality")


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------
def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DIARY_DIR, exist_ok=True)


def _now_iso():
    # Timestamp in the EATER'S current local time, not the server's. The person's
    # effective timezone (home, or a travel override while abroad) is resolved
    # centrally by person_registry so every tz-aware skill agrees. When --person
    # is given, THAT person's clock is used, not the speaker's. Degrade to
    # server time if that module is unavailable — a meal must never fail to log.
    try:
        import person_registry
        return person_registry.local_now_iso(CURRENT_PERSON)
    except Exception:
        return datetime.now().replace(microsecond=0).isoformat()


def _local_today():
    """Today's date on the EATER'S clock — the single source of 'today' for
    every command, so summaries and the backdate guard always agree."""
    return _now_iso()[:10]


def _print_header():
    """One-line trust anchor printed before every non-cron command's output.
    The agent must take person / now / today from HERE, never from memory."""
    now = _now_iso()
    try:
        wd = datetime.fromisoformat(now).strftime("%a")
    except Exception:
        wd = "?"
    print(f"[foodlog] person={_person_label()} | now={now} {wd} "
          f"(eater-local) | today={now[:10]}")
    note = _identity_note()
    if note:
        print(note)


def _infer_meal(ts_iso):
    try:
        h = datetime.fromisoformat(ts_iso).hour
    except Exception:
        return "snack"
    if 6 <= h < 12:
        return "breakfast"
    if 12 <= h < 16:
        return "lunch"
    if 16 <= h < 23:
        return "dinner"
    return "snack"


def _coerce_num(v):
    if v is None or v == "":
        return None
    try:
        n = float(v)
        return int(n) if n == int(n) else round(n, 1)
    except (TypeError, ValueError):
        return None


def _normalize(entry):
    """Fill defaults and coerce numerics. Never raises on partial input."""
    e = dict(entry) if isinstance(entry, dict) else {}
    # Assign UUID if missing (old entries won't have one)
    if "id" not in e or not e["id"]:
        e["id"] = str(uuid.uuid4())[:8]
    ts = str(e.get("ts") or "").strip() or _now_iso()
    # tolerate a trailing 'Z' or space-separated form
    ts = ts.replace("Z", "").replace(" ", "T", 1) if ts else _now_iso()
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        dt = datetime.now()
        ts = dt.replace(microsecond=0).isoformat()
    e["ts"] = ts
    e["date"] = dt.date().isoformat()
    # User-specified meal type takes priority; auto-infer only if not set
    if "meal" not in entry or not entry["meal"]:
        e["meal"] = _infer_meal(ts)
    else:
        e["meal"] = str(entry["meal"]).lower().strip()
    if not isinstance(e.get("items"), list):
        raw = e.get("items")
        e["items"] = [str(raw)] if raw else []
    for f in NUM_FIELDS:
        e[f] = _coerce_num(e.get(f))
    flags = e.get("flags")
    if isinstance(flags, str):
        flags = [flags]
    e["flags"] = [str(x).lower().strip() for x in flags] if isinstance(flags, list) else []
    e["confidence"] = str(e.get("confidence") or "low").lower().strip()
    e["note"] = str(e.get("note") or "").strip()
    e["photo"] = str(e.get("photo") or "").strip()
    # Wall-clock moment this entry was actually WRITTEN, on the eater's clock —
    # independent of the claimed "ts". A later audit can then tell a live log
    # from one backdated out of a stale session, which "ts" alone cannot show.
    e["logged_at"] = _now_iso()
    return e


def _append(entry):
    _ensure_dirs()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _append_diary(entry):
    path = os.path.join(DIARY_DIR, entry["date"] + ".md")
    new = not os.path.exists(path)
    tm = entry["ts"].split("T")[-1][:5]
    kcal = entry["kcal"]
    kcal_s = f"{kcal} kcal" if kcal is not None else "kcal ?"
    items = ", ".join(entry["items"]) if entry["items"] else entry["note"]
    q = entry["quality"]
    q_s = f" · quality {q}/10" if q is not None else ""
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(f"# Food diary — {entry['date']}\n\n")
        f.write(f"- **{tm}** ({entry['meal']}) — {items} — {kcal_s}{q_s}\n")


def _load():
    if not os.path.exists(LOG_PATH):
        return []
    out = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def _sum(entries, field):
    vals = [e.get(field) for e in entries if isinstance(e.get(field), (int, float))]
    return round(sum(vals), 1) if vals else 0


def _agg(entries):
    kcals = [e["kcal"] for e in entries if isinstance(e.get("kcal"), (int, float))]
    quals = [e["quality"] for e in entries if isinstance(e.get("quality"), (int, float))]
    flags = {}
    for e in entries:
        for fl in e.get("flags", []):
            flags[fl] = flags.get(fl, 0) + 1
    return {
        "n_meals": len(entries),
        "kcal": int(sum(kcals)) if kcals else 0,
        "protein_g": _sum(entries, "protein_g"),
        "carbs_g": _sum(entries, "carbs_g"),
        "fat_g": _sum(entries, "fat_g"),
        "fiber_g": _sum(entries, "fiber_g"),
        "avg_quality": round(statistics.mean(quals), 1) if quals else None,
        "n_missing_kcal": len(entries) - len(kcals),
        "flags": dict(sorted(flags.items(), key=lambda kv: -kv[1])),
    }


def _by_date(entries, date_str):
    return [e for e in entries if e.get("date") == date_str]


def _in_range(entries, start_date, end_date):
    return [e for e in entries if start_date <= e.get("date", "") <= end_date]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _phantom_notice(entries, today):
    """Say out loud that today is empty, and for how long.

    The failure this catches is the agent claiming «залогировал» without ever
    calling `add`: the reply looks normal, the journal never moved, and nobody
    notices until the day's totals are already wrong. An empty day is not proof
    of a lost write, but it is the only moment the tool can flag it in time.
    """
    if _by_date(entries, today):
        return ""
    last = max((e.get("ts", "") for e in entries), default="")
    if not last:
        return "NO ENTRIES TODAY. The journal is empty overall."
    try:
        gap = datetime.fromisoformat(_now_iso()) - datetime.fromisoformat(last)
        hours = int(gap.total_seconds() // 3600)
    except Exception:
        return f"NO ENTRIES TODAY. Last entry: {last}."
    return (
        f"NO ENTRIES TODAY ({today}). Last entry was {last} — {hours}h ago.\n"
        "If a meal was just logged this turn, it did NOT land: no `STORED:` "
        "receipt means `add` never ran. Run it once, now, without a \"ts\". "
        "Do NOT tell the user a meal is logged based on this output."
    )


def _fmt_day(date_str, entries, trailing=None):
    a = _agg(entries)
    lines = [f"=== {date_str} ==="]
    if not entries:
        lines.append("No meals logged.")
        return "\n".join(lines)
    lines.append(
        f"Meals: {a['n_meals']} | Calories: ~{a['kcal']} kcal"
        + (f" ({a['n_missing_kcal']} meal(s) without an estimate)" if a["n_missing_kcal"] else "")
    )
    lines.append(
        f"Protein {a['protein_g']}g | Carbs {a['carbs_g']}g | "
        f"Fat {a['fat_g']}g | Fiber {a['fiber_g']}g"
    )
    if a["avg_quality"] is not None:
        lines.append(f"Avg nutrient quality: {a['avg_quality']}/10")
    if a["flags"]:
        lines.append("Tags: " + ", ".join(f"{k} x{v}" for k, v in a["flags"].items()))
    lines.append(f"Meals on {date_str}:")
    for e in sorted(entries, key=lambda x: x.get("ts", "")):
        tm = e.get("ts", "").split("T")[-1][:5]
        kc = f"~{e['kcal']}kcal" if e.get("kcal") is not None else "kcal?"
        it = ", ".join(i.get("name", str(i)) if isinstance(i, dict) else str(i) for i in e.get("items", [])) or e.get("note", "")
        lines.append(f"  {tm} {e.get('meal','?')}: {it} ({kc})")
    if trailing:
        lines.append(
            f"Trailing 7-day average: ~{trailing['kcal_per_day']} kcal/day, "
            f"protein ~{trailing['protein_per_day']}g/day, "
            f"avg quality {trailing['quality'] if trailing['quality'] is not None else 'n/a'}/10."
        )
    return "\n".join(lines)


def _trailing(entries, end_date, days=7):
    end = datetime.fromisoformat(end_date).date()
    start = (end - timedelta(days=days - 1)).isoformat()
    window = _in_range(entries, start, end_date)
    active_days = len({e["date"] for e in window}) or 1
    a = _agg(window)
    return {
        "kcal_per_day": int(a["kcal"] / active_days),
        "protein_per_day": round(a["protein_g"] / active_days, 1),
        "quality": a["avg_quality"],
        "active_days": active_days,
    }


def _fmt_week(entries, today):
    end = datetime.fromisoformat(today).date()
    this_start = (end - timedelta(days=6)).isoformat()
    prev_end = (end - timedelta(days=7)).isoformat()
    prev_start = (end - timedelta(days=13)).isoformat()

    this_w = _in_range(entries, this_start, today)
    prev_w = _in_range(entries, prev_start, prev_end)
    if not this_w:
        return "NO_DATA"

    ta, pa = _agg(this_w), _agg(prev_w)
    t_days = len({e["date"] for e in this_w}) or 1
    p_days = len({e["date"] for e in prev_w}) or 1

    def per(a, days, field):
        return round(a[field] / days, 1)

    lines = [f"=== Week {this_start} .. {today} ==="]
    lines.append(f"Days with entries: {t_days}/7 | Total meals: {ta['n_meals']}")
    lines.append(f"Avg ~{int(ta['kcal']/t_days)} kcal/day (this week)")
    if prev_w:
        d = int(ta["kcal"] / t_days) - int(pa["kcal"] / p_days)
        arrow = "up" if d > 0 else ("down" if d < 0 else "flat")
        lines.append(f"  vs previous 7 days ~{int(pa['kcal']/p_days)} kcal/day -> {arrow} {abs(d)} kcal/day")
    lines.append(
        f"Avg macros/day: protein {per(ta,t_days,'protein_g')}g, "
        f"carbs {per(ta,t_days,'carbs_g')}g, fat {per(ta,t_days,'fat_g')}g, "
        f"fiber {per(ta,t_days,'fiber_g')}g"
    )
    if ta["avg_quality"] is not None:
        q_line = f"Avg nutrient quality: {ta['avg_quality']}/10"
        if prev_w and pa["avg_quality"] is not None:
            q_line += f" (prev week {pa['avg_quality']}/10)"
        lines.append(q_line)
    if ta["flags"]:
        lines.append("Most frequent tags: " + ", ".join(f"{k} x{v}" for k, v in list(ta["flags"].items())[:6]))
    # per-day kcal sparkline-ish breakdown
    lines.append("Per day:")
    for i in range(7):
        d = (datetime.fromisoformat(this_start).date() + timedelta(days=i)).isoformat()
        da = _agg(_by_date(this_w, d))
        if da["n_meals"]:
            lines.append(f"  {d}: {da['n_meals']} meals, ~{da['kcal']} kcal"
                         + (f", quality {da['avg_quality']}/10" if da["avg_quality"] is not None else ""))
        else:
            lines.append(f"  {d}: -")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def _guard_entry_date(entry, backdate_ok):
    """Refuse entries whose ts does not fall on the eater's CURRENT day.

    This is the barrier against the classic failure: a long-running chat
    session 'remembers' yesterday's date and logs this morning's coffee into
    yesterday evening. Returns None when the entry may be stored, else an
    error string (caller prints it and exits 3).
    """
    today = _local_today()
    d = entry["date"]
    if d == today:
        return None
    who = _person_label()
    if d > today:
        return (f"REFUSED: ts={entry['ts']} is in the FUTURE (today for "
                f"person '{who}' is {today}). Omit \"ts\" entirely - the tool "
                f"stamps the eater's local time itself.")
    if not backdate_ok:
        return (f"REFUSED: ts={entry['ts']} falls on {d}, but today for "
                f"person '{who}' is {today}.\n"
                f"- Meal eaten just now? OMIT \"ts\" - the tool stamps the "
                f"eater's local time itself.\n"
                f"- The USER explicitly said it was eaten on a past day? "
                f"Re-run with --backdate.\n"
                f"Never take the date from conversation memory; trust this "
                f"tool's clock.")
    return None


def cmd_add(args):
    raw = args.json
    if raw in (None, "-"):
        raw = sys.stdin.read()
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"ERROR: could not parse meal JSON: {e}", file=sys.stderr)
        return 2
    entry = _normalize(obj)
    err = _guard_entry_date(entry, getattr(args, "backdate", False))
    if err:
        print(err, file=sys.stderr)
        return 3

    before = _load()
    prev_last = max((e.get("date", "") for e in before), default="")

    _append(entry)
    try:
        _append_diary(entry)
    except Exception:
        pass  # diary is best-effort; never fail the log write on it

    _widget_push()

    all_entries = _load()
    day = entry["date"]
    if day > prev_last:
        prev = f"(previous entry was {prev_last})" if prev_last else "(journal was empty)"
        print(f"NEW DAY: first entry for {day} {prev}")
    if day != _local_today():
        print(f"BACKDATED: entry stored on {day}; today is {_local_today()}.")
    print("STORED:")
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    print(f"log: {LOG_PATH} | entries now: {len(all_entries)}")
    print()
    print(_fmt_day(day, _by_date(all_entries, day), _trailing(all_entries, day)))
    return 0


def cmd_correct(args):
    """Delete the last entry with matching meal type on today, then add the corrected one.

    The replacement JSON is parsed and validated BEFORE anything is deleted, so
    a malformed correction can never destroy the original entry.
    """
    all_entries = _load()
    meal_type = args.meal_type.lower().strip()

    # Parse + validate the replacement FIRST — nothing is deleted on error.
    raw = args.json
    if raw == "-":
        raw = sys.stdin.read()
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"ERROR: could not parse meal JSON (nothing was deleted): {e}",
              file=sys.stderr)
        return 2

    # Find the last entry with this meal type on today (eater-local today)
    today = _local_today()
    candidates = [e for e in all_entries if e.get("meal") == meal_type and e.get("date") == today]
    if not candidates:
        print(f"ERROR: no '{meal_type}' entries found for {today} "
              f"(person '{_person_label()}'). If the entry has a different "
              f"meal type, delete it by id and re-add instead.", file=sys.stderr)
        return 1

    target = candidates[-1]
    target_id = target.get("id")
    if not target_id:
        print(f"ERROR: entry has no ID, cannot delete", file=sys.stderr)
        return 1

    # Inherit meal type from target if not specified
    if not obj.get("meal"):
        obj["meal"] = meal_type

    entry = _normalize(obj)
    err = _guard_entry_date(entry, getattr(args, "backdate", False))
    if err:
        print(err + "\n(nothing was deleted)", file=sys.stderr)
        return 3

    # Rewrite log without the target entry (temp file + atomic replace, so a
    # crash mid-rewrite can never truncate the journal)
    _ensure_dirs()
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    tmp_path = LOG_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("id") == target_id:
                    continue  # skip — deleted
            except json.JSONDecodeError:
                pass
            f.write(line + "\n")
    os.replace(tmp_path, LOG_PATH)

    print(f"DELETED: id={target_id} [{meal_type}] at {target.get('ts', '')}")

    _append(entry)
    try:
        _append_diary(entry)
    except Exception:
        pass

    _widget_push()

    all_entries = _load()
    day = entry["date"]
    print("CORRECTED:")
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    print(f"log: {LOG_PATH} | entries now: {len(all_entries)}")
    print()
    print(_fmt_day(day, _by_date(all_entries, day), _trailing(all_entries, day)))
    return 0


def cmd_today(args):
    entries = _load()
    today = _local_today()
    print(_fmt_day(today, _by_date(entries, today), _trailing(entries, today)))
    notice = _phantom_notice(entries, today)
    if notice:
        print(notice)
    return 0


def cmd_day(args):
    entries = _load()
    print(_fmt_day(args.date, _by_date(entries, args.date), _trailing(entries, args.date)))
    return 0


def cmd_recent(args):
    entries = sorted(_load(), key=lambda e: e.get("ts", ""))[-args.n:]
    if not entries:
        print("No meals logged yet.")
        return 0
    for e in reversed(entries):
        kc = f"~{e['kcal']}kcal" if e.get("kcal") is not None else "kcal?"
        q = f" q{e['quality']}/10" if e.get("quality") is not None else ""
        it = ", ".join(i.get("name", str(i)) if isinstance(i, dict) else str(i) for i in e.get("items", [])) or e.get("note", "")
        print(f"{e.get('ts','')} [{e.get('meal','?')}] {it} ({kc}{q})")
    return 0


def cmd_week(args):
    entries = _load()
    today = _local_today()
    print(_fmt_week(entries, today))
    return 0


def cmd_report(args):
    entries = _load()
    today = _local_today()
    if args.period == "week":
        out = _fmt_week(entries, today)
        print(out if out else "NO_DATA")
        return 0
    day_entries = _by_date(entries, today)
    if not day_entries:
        print("NO_DATA")
        return 0
    print(_fmt_day(today, day_entries, _trailing(entries, today)))
    return 0


def main():
    p = argparse.ArgumentParser(description="Food log store + analytics")
    p.add_argument(
        "--person",
        default=None,
        help="Person key whose log to use (overrides $HERMES_PERSON). "
        "Every person is namespaced under persons/<key>/.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="Append one meal (JSON via --json or stdin)")
    a.add_argument("--json", default="-", help="Meal JSON, or '-' for stdin")
    a.add_argument("--backdate", action="store_true",
                   help="Allow a ts on a PAST day (only when the user explicitly "
                        "said the meal was eaten then)")
    a.set_defaults(fn=cmd_add)

    c = sub.add_parser("correct", help="Replace the last entry of a meal type on today")
    c.add_argument("meal_type", help="Meal type to correct (breakfast/lunch/dinner/snack/drink)")
    c.add_argument("--json", default="-", help="Corrected meal JSON, or '-' for stdin")
    c.add_argument("--backdate", action="store_true",
                   help="Allow the replacement's ts to fall on a past day")
    c.set_defaults(fn=cmd_correct)

    sub.add_parser("today", help="Today's summary").set_defaults(fn=cmd_today)

    d = sub.add_parser("day", help="A specific day's summary")
    d.add_argument("date", help="YYYY-MM-DD")
    d.set_defaults(fn=cmd_day)

    r = sub.add_parser("recent", help="Last N meals")
    r.add_argument("n", nargs="?", type=int, default=6)
    r.set_defaults(fn=cmd_recent)

    sub.add_parser("week", help="7-day trend vs prior 7 days").set_defaults(fn=cmd_week)

    rep = sub.add_parser("report", help="Cron-friendly block; NO_DATA if empty")
    rep.add_argument("period", nargs="?", choices=["today", "week"], default="today")
    rep.set_defaults(fn=cmd_report)

    args = p.parse_args()
    if getattr(args, "person", None) is not None:
        _set_person(args.person)
    if args.cmd != "report":
        # report is consumed by cron prompts whose contract is "exactly
        # NO_DATA when empty" — everything else gets the trust-anchor header.
        _print_header()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
