# foodlog.py - tooling reference

## Commands

Global flag: `--person <key>` (BEFORE the subcommand) - operate on that person's journal, widget and timezone. Default: the session speaker.

| Command | Description |
|---|---|
| `foodlog.py add --json -` | Append one meal (JSON via stdin or `--json '<obj>'`). `--backdate` allows a `ts` on a past day |
| `foodlog.py correct <meal_type> --json -` | Delete today's last entry of that meal type, append the replacement (validated BEFORE deletion) |
| `foodlog.py today` | Today's summary + trailing 7-day average |
| `foodlog.py day YYYY-MM-DD` | A specific day's summary |
| `foodlog.py recent [N]` | Last N meals (default 6), newest first |
| `foodlog.py week` | Last 7 days vs prior 7 days |
| `foodlog.py report [today\|week]` | Cron-friendly block; prints exactly `NO_DATA` when empty |

## Output contract (trust anchors for the agent)

- `[foodlog] person=... | now=... Fri (eater-local) | today=...` - first line of every command except `report`. THE authoritative person, clock and date for this call; always beats any date/count remembered from conversation.
- `WARNING:` / `NOTE:` line right under the header - identity trouble: the gateway session has NO speaker id (or one the registry cannot resolve), so the call operates on the PRIMARY journal; or the speaker is unregistered (`persons/unknown`). Never ignore it: route with `--person <key>` if the food is not that person's, and tell the owner the session lost its identity.
- `NEW DAY: first entry for <date> ...` - this add opened a new day. Narrate totals as a fresh day.
- `BACKDATED: entry stored on <date>; today is <date>.` - only with `--backdate`.
- `REFUSED` (exit 3) - the JSON carried a `ts` on a non-today date without `--backdate`, or a future `ts`. The tool is right: omit `ts` for a meal eaten now; use `--backdate` only when the user explicitly named a past day.
- `STORED:` / `CORRECTED:` + `log: <path> | entries now: N` - the write receipt (person-routed path + journal size). This is complete verification of the write; nothing needs re-checking later.
- Exit codes: 0 ok · 1 target not found (`correct`) · 2 bad JSON (nothing deleted) · 3 date guard refused.
- `report` prints NO header and exactly `NO_DATA` when the period is empty - cron prompts turn that into [SILENT]. Do not change this contract.

## Data layout (per person, append-only)

```
~/.hermes/data/food/persons/<key>/
    log.jsonl                 # source of truth, one meal per line, APPEND-ONLY
    diary/YYYY-MM-DD.md       # human-readable diary (best-effort)
    photos/                   # optional saved photos
```

- `add` is a pure append; `correct` rewrites via temp file + atomic replace. Nothing ever overwrites the journal wholesale; multiple sessions may append concurrently.
- Timestamps are stamped in the EATER's local time via `person_registry` (travel overrides included); `--person` uses THAT person's clock.
- Test overrides: `$FOODLOG_DIR` re-points storage, `$FOODLOG_NO_WIDGET=1` suppresses the widget push.

## Weight & the food↔weight link (Home Assistant / Withings)

| Command | Description |
|---|---|
| `weight.py --person <key>` | Weight + body composition: up to 8 most recent weigh-ins over 7 days, Δ, kg/week trend, goal + ETA (if that person has a goal sensor) |
| `weight.py --person <key> --json` | Same, machine-readable |
| `weight.py --person <key> --selftest` | Shape only (counts, which sensors resolved) - prints no values; safe for debugging |
| `food_weight_digest.py --person <key> --period day\|week` | What the digest crons run: food + weight + implied TDEE. `NO_DATA` when no food logged |
| `health_weekly_digest.py` | Garmin week-over-week + weight; feeds `health-weekly-digest` |

- Entity names derive from the person key; missing sensors are skipped (same code serves every person in `users.json`).
- Credentials: `HASS_URL` / `HASS_TOKEN` in `~/.hermes/.env` (NOT `HA_TOKEN`, NOT `~/.env` - that mistake produced a silently empty report).
- The weigh-in window auto-widens to 14 days if the week holds fewer than 2 readings, and says so.
- Implied TDEE (`avg intake - Δkg × 7700 / days`) is only emitted with ≥4 food days AND ≥4 days between weigh-ins.

## Repair recipes (manual surgery)

Delete by **`id`** (8-char UUID from the STORED receipt), never by timestamp - a genuine entry can share a timestamp.

**Wrong person:**
```bash
grep -v '<id>' ~/.hermes/data/food/persons/<wrong>/log.jsonl > /tmp/fx && mv /tmp/fx ~/.hermes/data/food/persons/<wrong>/log.jsonl
python3 ~/.hermes/scripts/foodlog.py --person <right> add --json - <<'JSON' ... JSON
python3 ~/.hermes/scripts/foodlog.py --person <right> today   # verify
python3 ~/hermes-widgets/hooks/food_widget_push.py --user <wrong>   # BOTH widgets
python3 ~/hermes-widgets/hooks/food_widget_push.py --user <right>
```

**Wrong meal type** (`correct` will not match it): delete by id as above, re-add with the right `meal` and the real same-day `ts` (a past day needs `--backdate`). Push that person's widget.

**Cross-date mistakes:** check `day <date>` on BOTH days before and after the surgery; a real meal may exist on the wrong day at the same time.

**After ANY manual log.jsonl edit** the automatic widget push does NOT fire (foodlog.py was never invoked) - always run `food_widget_push.py --user <key>` for every journal you touched.

## Widget: push sent but the phone is stale (known FCM issue)

First check `~/hermes-widgets/push.log`: every automatic push appends one status line (`<ts> user=<key> exit=<rc> published=yes|no fcm=...`). No line at meal time = the hook never fired (the meal was not logged at all, or was logged for another person - check `today` for the right person). A `fcm=sent` line while the phone is stale = phone-side delivery:
1. Ask the user to pull down on the widget (most common fix).
2. Do Not Disturb on the phone blocks pushes.
3. Restart the widget app.
4. Data actually wrong (missing meal)? Check `foodlog.py today` first - the log itself may be missing the entry (phantom logging: the reply claimed "залогировал" but `add` never ran).

«Виджет не обновился» → first verify the log is correct, then retry the push, then pull-to-refresh.

## Design decisions

- **Delete, not supersede.** `correct` removes the old entry and appends the replacement; no stale rows accumulate.
- **8-char UUID `id`** per entry (auto-generated).
- **Widget push is automatic** after every `add`/`correct` (fire-and-forget; a widget hiccup never fails a meal write). Do not call it manually except after manual file edits.
- **Date guard at the tool level.** A `ts` on a non-today date is refused without `--backdate` - conversation-stale dates cannot silently land in the journal.
- **Pure stdlib, JSONL, no cloud.**
