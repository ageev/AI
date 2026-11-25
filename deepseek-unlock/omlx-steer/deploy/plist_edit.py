"""install / uninstall edits of the oMLX launchd plist (only two keys change).

    plist_edit.py install PLIST STATE ROOT DATA SWEEP
    plist_edit.py uninstall STATE
"""
import json
import plistlib
import sys
from pathlib import Path

ENV_KEYS = ("OMLX_STEER_PROFILES", "OMLX_STEER_SWEEP")


def load(path):
    with open(path, "rb") as f:
        return plistlib.load(f)


def save(path, data):
    tmp = Path(str(path) + ".omlx-steer-tmp")
    with open(tmp, "wb") as f:
        plistlib.dump(data, f)
    tmp.replace(path)


def install(plist, state, root, data_dir, sweep):
    if Path(state).exists():
        sys.exit(f"already installed (state in {state}); run uninstall.sh first")
    data = load(plist)
    args = list(data["ProgramArguments"])
    if any("omlx-steer" in a for a in args):
        sys.exit("plist already points at omlx-steer")
    env = data.get("EnvironmentVariables", {})
    Path(state).write_text(json.dumps({
        "plist": str(plist),
        "ProgramArguments": args,
        "env_present": {k: k in env for k in ENV_KEYS},
    }, indent=1))
    data["ProgramArguments"] = [f"{root}/deploy/omlx-python.sh", f"{root}/deploy/launch.py", *args[1:]]
    env = dict(env)
    env["OMLX_STEER_PROFILES"] = f"{data_dir}/profiles"
    env["OMLX_STEER_SWEEP"] = "1" if sweep == "1" else "0"
    data["EnvironmentVariables"] = env
    save(plist, data)
    print("ProgramArguments:", data["ProgramArguments"])


def uninstall(state):
    st = json.loads(Path(state).read_text())
    data = load(st["plist"])
    data["ProgramArguments"] = st["ProgramArguments"]
    env = dict(data.get("EnvironmentVariables", {}))
    for key in ENV_KEYS:
        if not st["env_present"].get(key):
            env.pop(key, None)
    data["EnvironmentVariables"] = env
    save(st["plist"], data)
    Path(state).unlink()
    print("restored ProgramArguments:", data["ProgramArguments"])


if __name__ == "__main__":
    if sys.argv[1] == "install":
        install(*sys.argv[2:7])
    else:
        uninstall(sys.argv[2])
