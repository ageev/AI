"""Decode speed and MTP acceptance, stock vs steered, at several concurrencies.

    speed.py URL KEYFILE SERVER_LOG MODEL [MODEL ...] [--conc 1,8] [--tokens 768] [--rounds 2]

Each round sends `conc` identical-shape requests (different prompts, greedy,
no thinking) and reports per-request decode tok/s (completion tokens / wall
time minus TTFT, streaming) and the MTP acceptance logged in the same window.
"""
import argparse
import json
import re
import statistics
import threading
import time
import urllib.request

PROMPTS = [
    "Write a detailed essay about the history of bridges, from stone arches to suspension bridges.",
    "Explain how a modern CPU executes instructions, covering pipelines, caches and branch prediction.",
    "Describe the water cycle in depth, including evaporation, condensation and groundwater.",
    "Write a long story about a lighthouse keeper who finds a message in a bottle.",
    "Explain the causes and consequences of the industrial revolution in Europe.",
    "Describe how vaccines train the immune system, step by step.",
    "Write a guide to growing tomatoes on a balcony, from seed to harvest.",
    "Explain the basics of double-entry bookkeeping with several worked examples.",
]
ACCEPT = re.compile(r"MTP\[\d+\] .*?accept=(\d+)/(\d+)")


def stream(url, key, model, prompt, tokens, out, i):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json", "authorization": f"Bearer {key}"})
    t0, first, n = time.time(), None, 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or {}
            if usage.get("completion_tokens"):
                n = usage["completion_tokens"]
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                if first is None:
                    first = time.time()
    end = time.time()
    # Tokens after the first one, over the time after the first one arrived.
    out[i] = (max(n - 1, 0), end - (first or end))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("keyfile")
    ap.add_argument("log")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--conc", default="1,8")
    ap.add_argument("--tokens", type=int, default=768)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    key = open(a.keyfile).read().strip()
    rows = []
    for conc in [int(c) for c in a.conc.split(",")]:
        for model in a.models:
            rates, acc, tot = [], 0, 0
            for rnd in range(a.rounds):
                start = open(a.log, "rb").seek(0, 2)
                out = {}
                threads = [threading.Thread(target=stream, args=(a.url, key, model, PROMPTS[(rnd * conc + i) % len(PROMPTS)], a.tokens, out, i))
                           for i in range(conc)]
                [t.start() for t in threads]
                [t.join() for t in threads]
                rates += [n / dt for n, dt in out.values() if dt > 0]
                time.sleep(1.0)
                with open(a.log, "rb") as f:
                    f.seek(start)
                    for m in ACCEPT.finditer(f.read().decode(errors="replace")):
                        acc += int(m.group(1))
                        tot += int(m.group(2))
            row = {"model": model, "conc": conc, "tok_s_per_req": round(statistics.median(rates), 2),
                   "mtp_accept": round(acc / tot, 3) if tot else None}
            rows.append(row)
            print(json.dumps(row), flush=True)
    base = {r["conc"]: r["tok_s_per_req"] for r in rows if r["model"] == a.models[0]}
    for r in rows:
        r["vs_first"] = round(r["tok_s_per_req"] / base[r["conc"]], 3)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
