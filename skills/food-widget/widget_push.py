#!/usr/bin/env python3
"""Render a widget slot and publish it to the static web share.

Usage:
  widget_push.py SLOT [--json FILE] [--local-only] [--out DIR]

Payload is read from --json or stdin. Renders every bucket/theme from
config.toml, writes meta.json + manifest.json, then uploads via SMB
(unless --local-only). Errors go straight to the console with the cause.
"""
import argparse
import json
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from render.slots import REGISTRY  # noqa: E402


def _atomic_write(path: Path, data: bytes):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slot", choices=sorted(REGISTRY))
    ap.add_argument("--json", help="payload file (default: stdin)")
    ap.add_argument("--local-only", action="store_true", help="render only, no upload")
    ap.add_argument("--out", default=str(ROOT / "out"))
    args = ap.parse_args()

    cfg = tomllib.loads((ROOT / "config.toml").read_text())
    raw = Path(args.json).read_text() if args.json else sys.stdin.read()
    payload = json.loads(raw)
    payload.setdefault("updated",
                       datetime.now().astimezone().isoformat(timespec="seconds"))

    entry = REGISTRY[args.slot]
    mod = entry["module"]
    title = entry.get("title") or mod.TITLE
    ttl = entry.get("ttl_minutes") or mod.TTL_MINUTES
    payload.setdefault("_title", title)
    outdir = Path(args.out) / args.slot
    outdir.mkdir(parents=True, exist_ok=True)

    images = {}
    for bucket, (w, h) in cfg["render"]["buckets"].items():
        for theme in cfg["render"]["themes"]:
            img = mod.render(payload, w, h, theme)
            name = f"{bucket}-{theme}.png"
            from io import BytesIO
            buf = BytesIO()
            img.save(buf, "PNG")
            _atomic_write(outdir / name, buf.getvalue())
            images.setdefault(bucket, {})[theme] = name
            print(f"rendered {args.slot}/{name} ({w}x{h})")

    meta = {
        "slot": args.slot,
        "title": title,
        "updated": payload["updated"],
        "ttl_minutes": ttl,
        "images": images,
        "v": int(time.time()),
    }
    # tap_url precedence: per-slot config -> [slots._default] -> payload.
    # Project decision 2026-07-07: default opens the Hermes conversation
    # (tg://resolve); topic deep links do not work on the user's client.
    slots_cfg = cfg.get("slots", {})
    tap_url = (slots_cfg.get(args.slot, {}).get("tap_url")
               or slots_cfg.get("_default", {}).get("tap_url")
               or payload.get("tap_url") or "")
    if tap_url == "none":
        # per-slot sentinel: deliberately NO tap action (tap = refresh);
        # an empty value still inherits [slots._default] as before.
        tap_url = ""
    if tap_url:
        meta["tap_url"] = tap_url
    _atomic_write(outdir / "meta.json",
                  json.dumps(meta, ensure_ascii=False, indent=1).encode())

    manifest_path = Path(args.out) / "manifest.json"
    manifest = {"schema": 1, "slots": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest["slots"][args.slot] = {"title": title, "updated": payload["updated"]}
    manifest["generated"] = payload["updated"]
    _atomic_write(manifest_path,
                  json.dumps(manifest, ensure_ascii=False, indent=1).encode())

    if args.local_only:
        print("local-only: upload skipped")
        return 0

    from upload_smb import Publisher
    pub = Publisher(cfg)
    pub.publish_slot(args.slot, outdir, manifest_path)
    print(f"published: {pub.public_hint(args.slot)}")

    try:
        from fcm_send import send_slot_update
        name = send_slot_update(args.slot, meta["v"], meta["updated"])
        print(f"FCM push sent ({name.rsplit('/', 1)[-1]})")
    except Exception as e:
        print(f"FCM push skipped: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
