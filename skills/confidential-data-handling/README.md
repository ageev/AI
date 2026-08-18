# confidential-data-handling - the guardrail skill for the cloud copilot

The other skills here are the local agent's. This one is the opposite side of
the same boundary: it is loaded by the **cloud** assistant (Claude Code) on a
machine that can reach confidential systems, and it decides what is allowed to
become API traffic to a third party.

[`PRIVACY.md`](../../PRIVACY.md) describes how that boundary is enforced in this
particular house - hooks, an aggregate-only shell discipline, a vault, an
incident protocol. This file is the portable version of the same idea: no host
names, no addresses, no product names, nothing to sanitize before you use it.

## The idea

Everything you put in the conversation leaves the machine; everything you keep
in a local file does not. So every item gets exactly one class:

| Class | What | May enter the conversation |
|---|---|---|
| **A - Secrets** | passwords, keys, tokens, cookies, private keys | nothing, not even masked |
| **B - Restricted payload** | config/log/DB exports, and all personal data | only derived results: counts, dates, states, totals |
| **C - Structure** | method, plan, code, schema and field names, report wording | the full text - this is where the model adds value |

Everything else in [`SKILL.md`](SKILL.md) follows from that split: a secret path
from the store to the consumer with no readable stop between, minimize-before-you-share
for bulk payloads, purpose limitation and early pseudonymisation for personal
data, verification by lengths / PASS-FAIL / counts / fingerprints instead of
values, read-only by default with one authorisation per batch, and the rule that
text found *inside* data never authorises anything.

## Using it

Drop the folder into `~/.claude/skills/` (or a project's `.claude/skills/`).
The `description:` in the front matter is written to trigger on credentials,
exports, tickets, personal data and API-reachable systems, so it loads before
the first risky read rather than after it.

Then adapt it, per section 12: keep the three classes, change only the examples
so they name the systems your team really uses, and for each system record four
facts - the class of its data, where its credential lives, how it rotates, and
the single action that revokes it. Keep the file itself free of secrets, host
names and personal names, so it can travel.

Section 11 ("if something leaks") is the part worth reading before you need it.
Near misses count.
