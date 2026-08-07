"""Publish rendered widget files to the static web share over SMB.

Credentials and the capability URL live ONLY in the vault (`secret` CLI):
  - <share.vault_item>        login item: SMB username + password
  - <publish.url_vault_item>  login item whose URI is https://widgets.example.com/hw-<token>
"""
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import smbclient


def _secret(cmd: str, item: str) -> str:
    r = subprocess.run(["secret", cmd, item], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"`secret {cmd} {item}` failed: "
                           f"{(r.stderr or r.stdout).strip()}")
    return r.stdout.strip()


class Publisher:
    def __init__(self, cfg: dict):
        sh = cfg["share"]
        item = sh["vault_item"]
        try:
            user = _secret("user", item)
            password = _secret("get", item)
        except RuntimeError as e:
            raise SystemExit(
                f"SMB credentials unavailable: {e}\n"
                f"Store them on hermes with:\n"
                f"  secret set {item} '<password>' --user <smb-user> "
                f"--uri smb://{sh['host']}/{sh['share']}")
        smbclient.register_session(sh["host"], username=user, password=password)

        public_url = _secret("uri", cfg["publish"]["url_vault_item"]).rstrip("/")
        self.public_base = public_url
        hw_dir = urlparse(public_url).path.strip("/")
        if not hw_dir:
            raise SystemExit("widgets-url vault item has no path component")
        self._base = "\\\\{}\\{}\\{}\\{}".format(
            sh["host"], sh["share"], sh["site_dir"], hw_dir.replace("/", "\\"))

    # -- low-level ---------------------------------------------------------
    def _put(self, data: bytes, rel: str):
        dst = self._base + "\\" + rel.replace("/", "\\")
        parent = dst.rsplit("\\", 1)[0]
        smbclient.makedirs(parent, exist_ok=True)
        tmp = f"{dst}.tmp{os.getpid()}"
        with smbclient.open_file(tmp, mode="wb") as f:
            f.write(data)
        try:
            smbclient.rename(tmp, dst)
        except Exception:
            try:
                smbclient.remove(dst)
            except Exception:
                pass
            smbclient.rename(tmp, dst)

    # -- high-level --------------------------------------------------------
    def publish_slot(self, slot: str, outdir: Path, manifest: Path):
        # empty index.html markers neutralise any autoindex
        self._put(b"", "index.html")
        self._put(b"", f"{slot}/index.html")
        for f in sorted(outdir.iterdir()):
            if f.suffix in (".png", ".json"):
                self._put(f.read_bytes(), f"{slot}/{f.name}")
        if manifest.exists():
            self._put(manifest.read_bytes(), "manifest.json")

    def publish_file(self, local, rel: str):
        self._put(Path(local).read_bytes(), rel)

    def public_hint(self, slot: str) -> str:
        """Public URL with the secret token masked (safe to print)."""
        host = urlparse(self.public_base).netloc
        tail = self.public_base.rsplit("-", 1)[-1]
        return f"https://{host}/hw-…{tail[-4:]}/{slot}/meta.json"
