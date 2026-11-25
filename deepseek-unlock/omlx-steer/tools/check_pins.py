"""Exit 1 unless every pinned oMLX function matches pins.json (no server needed)."""
import importlib
import json
import sys

from omlx._version import __version__
from omlx_steer import PINS_FILE, SUPPORTED_OMLX, source_hash

if __version__ not in SUPPORTED_OMLX:
    sys.exit(f"oMLX {__version__} not supported (pinned {SUPPORTED_OMLX})")
bad = []
for module_name, names in json.loads(PINS_FILE.read_text()).items():
    module = importlib.import_module(module_name)
    bad += [f"{module_name}:{n}" for n, h in names.items() if source_hash(module, n) != h]
if bad:
    sys.exit("changed upstream: " + ", ".join(bad))
print(f"pins ok for oMLX {__version__}")
