"""Start oMLX with omlx-steer installed: ``launch.py <omlx-cli args>``.

Equivalent to ``omlx-cli <args>`` plus the steering hooks. Environment:
  OMLX_STEER_PROFILES  profiles directory (default ~/omlx-steer-data/profiles)
  OMLX_STEER_SWEEP=1   enable sweep profiles and request overrides
"""
import logging
import runpy
import sys

# Own handler: oMLX configures the root logger later; propagating would print
# every steering line twice.
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
_log = logging.getLogger("omlx_steer")
_log.addHandler(_handler)
_log.setLevel(logging.INFO)
_log.propagate = False

import omlx_steer  # noqa: E402

omlx_steer.install()
sys.argv = ["omlx.cli", *sys.argv[1:]]
runpy.run_module("omlx.cli", run_name="__main__", alter_sys=True)
