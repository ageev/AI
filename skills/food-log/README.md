# food-log - meal-photo calorie tracking for a local agent

The most used skill of the local agent: send a photo of your plate to the
Telegram bot, the local VLM estimates it, a deterministic Python tool stores it
and does every bit of arithmetic, and your phone widget refreshes. Everything
stays on the LAN.

## Design

- **The model estimates, Python counts.** A local 27B-class model is good at
  "this is chicken, rice and salad, ≈620 kcal" and terrible at summing 30 rows.
  So [`scripts/foodlog.py`](scripts/foodlog.py) owns all math (day totals,
  week-over-week, trailing averages) and the agent only narrates its output.
- **Append-only JSONL per person** (`~/.hermes/data/food/persons/<key>/log.jsonl`),
  multi-writer safe, plus a human-readable Markdown diary. No database.
- **Anti-hallucination rails, all born from real incidents** (see
  [`SKILL.md`](SKILL.md)): a date guard that REFUSES stale timestamps from
  long-running sessions, `STORED:` write receipts, a `[foodlog]` header the
  agent must trust over its own memory, phantom-log detection, `correct`
  that validates before deleting.
- **Multi-user:** a meal belongs to whose food it is, not who is typing;
  `--person` routes journal, widget and timezone together.

## Files

- [`SKILL.md`](SKILL.md) - the agent-facing skill (hermes-agent format)
- [`references/estimation.md`](references/estimation.md) - portion anchors, calorie ready-reckoner, quality rubric, flags vocabulary
- [`references/foodlog-tooling.md`](references/foodlog-tooling.md) - command/output contract, repair recipes, widget troubleshooting
- [`scripts/foodlog.py`](scripts/foodlog.py) - the store + analytics tool (pure stdlib)

## Omitted host-specific parts

`person_registry.py` (identity + per-person timezones; `foodlog.py` degrades
gracefully without it), `weight.py` / `food_weight_digest.py` (pull weight from
Home Assistant / Withings), cron digest wrappers. Person keys are sanitized to
`alice` / `bob`.

The widget half of the pipeline: [`../food-widget/`](../food-widget/).
