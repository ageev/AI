"""Collect last-token activations for direction finding through a live server.

    capture.py URL KEYFILE DATASETS_DIR [--sets harm_train,harmless_train,...] [--workers 4]

Every prompt is sent once per mode (think / nothink) to the capture profile
with ``max_tokens 1``; the server writes <set>.<mode>.<index>.npz into the
profile's capture dir. Existing files are skipped, so the run can resume.
"""
import argparse
import json
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SETS = {
    "harm_train": ("prompts.json", "harm_train"),
    "harmless_train": ("prompts.json", "harmless_train"),
    "advice_train": ("advice.json", "train"),
    "reference_train": ("reference.json", "train"),
}


def prompts(ds_dir, name):
    file, key = SETS[name]
    items = json.loads((Path(ds_dir) / file).read_text())[key]
    return [i["prompt"] if isinstance(i, dict) else i for i in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("keyfile")
    ap.add_argument("datasets")
    ap.add_argument("--capture-dir", default=str(Path.home() / "omlx-steer-data/profiles/capture"))
    ap.add_argument("--model", default="deepseek-capture")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--effort", type=int, default=50)
    a = ap.parse_args()
    key = open(a.keyfile).read().strip()
    jobs = []
    for name in a.sets.split(","):
        for i, text in enumerate(prompts(a.datasets, name)):
            for mode in ("think", "nothink"):
                tag = f"{name}.{mode}.{i:04d}"
                if not (Path(a.capture_dir) / f"{tag}.npz").exists():
                    jobs.append((tag, text, mode))
    print(f"{len(jobs)} captures to run", flush=True)
    done, lock, errors = [0], threading.Lock(), []

    def run(job):
        tag, text, mode = job
        body = {"model": a.model, "messages": [{"role": "user", "content": text}], "max_tokens": 1,
                "temperature": 0, "chat_template_kwargs": {"enable_thinking": mode == "think"},
                "omlx_steer": {"tag": tag}}
        if mode == "think":
            body["reasoning_effort"] = a.effort
        req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"content-type": "application/json", "authorization": f"Bearer {key}"})
        try:
            urllib.request.urlopen(req, timeout=600).read()
        except Exception as exc:
            errors.append((tag, str(exc)))
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"{done[0]}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(run, jobs))
    print(f"done, {len(errors)} errors", errors[:3])
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
