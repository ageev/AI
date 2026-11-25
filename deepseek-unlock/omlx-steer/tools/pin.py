"""Write omlx_steer/pins.json from the installed oMLX sources.

Run only after reviewing that the hook assumptions still hold for that oMLX
version (see docs/DESIGN.md, "Hook points").
"""
import importlib
import json

from omlx_steer import PINS_FILE, source_hash
from omlx_steer.adapters import deepseek_v41
from omlx_steer import integration

pins = {}
for table in (integration.PINNED, deepseek_v41.PINNED):
    for module_name, names in table.items():
        module = importlib.import_module(module_name)
        pins.setdefault(module_name, {})
        for name in names:
            pins[module_name][name] = source_hash(module, name)
PINS_FILE.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
print(f"wrote {PINS_FILE}: {sum(len(v) for v in pins.values())} pins")
