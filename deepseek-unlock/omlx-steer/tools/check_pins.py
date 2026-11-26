"""Exit 1 unless every pinned oMLX function matches pins.json (no server needed).

    check_pins.py [--any-version]

--any-version skips the version gate and lists every pinned function that
differs in the installed oMLX: the porting checklist for a new release.
"""
import importlib
import json
import sys

from omlx._version import __version__
from omlx_steer import PINS_FILE, SUPPORTED_OMLX, source_hash

if __version__ not in SUPPORTED_OMLX and "--any-version" not in sys.argv:
    sys.exit(f"oMLX {__version__} not supported (pinned {SUPPORTED_OMLX})")
bad = []
for module_name, names in json.loads(PINS_FILE.read_text()).items():
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        bad.append(f"{module_name} (module missing: {exc})")
        continue
    for name, pinned in names.items():
        try:
            changed = source_hash(module, name) != pinned
        except (AttributeError, OSError, TypeError) as exc:
            bad.append(f"{module_name}:{name} (missing: {exc})")
            continue
        if changed:
            bad.append(f"{module_name}:{name}")
if bad:
    sys.exit(f"changed upstream in oMLX {__version__}:\n  " + "\n  ".join(bad))
print(f"pins ok for oMLX {__version__}")
