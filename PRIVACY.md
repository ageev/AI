EN | [RU](PRIVACY.ru.md)

# Privacy boundaries: cloud architect, local hands

Two AIs with different trust levels run this setup:

- **Cloud copilot** (Claude Code over SSH): designs the architecture, writes and reviews the agent's skills, fixes configs, debugs hosts. Everything it reads becomes API traffic to a third party, so it is treated as an untrusted reader.
- **Local agent** ([hermes-agent](https://github.com/NousResearch/hermes-agent) + a local LLM on the GB10): runs the household - Telegram, smart home, health dashboards, family logs. Inference is local, its context never leaves the LAN, so it may read everything.

The asymmetry is the whole design: **the mentor teaches the agent how to work but never reads the data the agent works with.** Whatever you think of cloud vendors' data handling, the clean answer is that private bytes never enter a cloud prompt at all.

## What the cloud copilot never reads

- agent memory and profiles (`MEMORY.md`, `memories/`, `sessions/`, `logs/`) - the family's life in text form
- chat transcripts (Telegram)
- `config.yaml` and `.env` - tokens, keys, and even URLs (a Home Assistant URL is a geolocation)
- Home Assistant state in any form: presence, device lists, wearable health metrics
- health databases, food logs, family ledgers, photo folders
- sneaky side channels: a bot-API data dir where subdirectory names ARE the bot tokens; DEBUG logs that dump auth headers. Listing a directory can already be a leak.

## How it is enforced (not just promised)

1. **Rules load automatically.** A `UserPromptSubmit` hook detects agent-related tasks and injects the rules file into the session before work starts. The model must acknowledge with a sha-stamped marker as its first output line, so the human gets two independent signals (hook banner + model ack) that the rules are active. Rules that rely on someone remembering them fail; hooks do not.
2. **Aggregate-only shell discipline.** On any file that may hold personal data, only non-printing commands are allowed: `wc -l`, `grep -c`, `grep -ilE` (filenames only), `grep -oE` (matched token only), `sha256sum`. Never `cat`/`head`/`tail`/`less`, never line-printing `grep`/`awk`/`sed`. Names and structure (`ls`, `find`) are fine; contents are not.
3. **Config writes without reads.** Agent config changes go through `hermes config set KEY VALUE`; verification is `grep -oE` for the expected token. The config file is never opened - it holds secrets.
4. **Delegation instead of access.** When a task genuinely needs private data, the copilot writes a precise question and the local agent answers with sanitized output: aggregates, statuses, entity names without values. "Did the automation fire today, yes/no" instead of a location-history dump.
5. **Secrets live in a vault.** vaultwarden + `bw serve` + a small `secret` CLI on the agent host; the agent fetches its own credentials at runtime. The copilot handles secret *names*, never values. On the NAS, configs render from an env file into tmpfs, so no plaintext secrets sit in the compose tree either.
6. **Violations are loud.** Accidentally reading something private is an *incident*: stop immediately, report to the human, no quiet finishing. The rules got their teeth from exactly one such incident - a genomics session leaked three genotype values into cloud context via casual `head`/`awk` peeks. Result: a permanent aggregate-only rule for genotype files and the incident protocol above.

## The same pattern beyond the agent

- Genome project: the copilot is architect-only; pipelines are designed and reviewed in the abstract, genotype values never enter the context (only `wc -l`, `grep -c`, checksums, `bcftools stats`).
- Health DB: schemas and record counts are fair game, field values are not.
- Anything new defaults to deny: if a file *might* be private, treat it as private.

## Why this works

- The mentor needs the *shape* of the data, not the data: schemas, counts, exit codes, error messages. That is almost always enough to design and debug.
- Deny-by-default beats redaction: do not read-then-scrub, just do not read.
- Each side does what it is best at: the frontier model designs, the local model touches.
