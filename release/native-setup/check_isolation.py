"""Verify the actual macOS sandbox profile's file-read and socket denials."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def check(profile, denied_root, output):
    # Probe only the loopback discard port; no remote DNS/server is contacted.
    target = str(Path(denied_root).resolve() / "README.md")
    worker = '''import errno,json,socket
try:
    open(TARGET, "rb").read(1)
except PermissionError:
    file_read_denied=True
else:
    raise RuntimeError("Original project read was not denied")
sock=socket.socket()
try:
    code=sock.connect_ex(("127.0.0.1", 9))
finally:
    sock.close()
if code not in (errno.EPERM,errno.EACCES):
    raise RuntimeError("Loopback outbound connection was not denied: " + str(code))
print(json.dumps({"file_read_denied":file_read_denied,"socket_connect_denied":True,"socket_error_code":code,"network_body_bytes":0}))
'''.replace("TARGET", repr(target))
    python = Path(sys.base_prefix) / "bin/python3"
    if not python.is_file():
        raise RuntimeError("Persistent base Python executable unavailable for guard probe")
    result = subprocess.run(["/usr/bin/sandbox-exec", "-f", str(Path(profile).resolve()), str(python.resolve()), "-c", worker], cwd="/", capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Sandbox guard verification failed: " + result.stderr[-2000:])
    receipt = {"schema": "akash-native-isolation-guard-check-v1", "checked_at": datetime.now(timezone.utc).isoformat(), "status": "passed", "profile": str(Path(profile).resolve()), "denied_root": str(Path(denied_root).resolve()), "probe_scope": "Actual denied original README read and loopback outbound socket; no remote server request", **json.loads(result.stdout)}
    with Path(output).open("x") as stream:
        json.dump(receipt, stream, indent=2); stream.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--denied-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.profile, args.denied_root, args.out), indent=2))
