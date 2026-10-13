"""Start pinned local Qwen on CPU, or ask its verified fact-selection assistant.

Install runtime: brew install llama.cpp
Download the official pinned GGUF in MODEL to data/copilot-model, verify its hash.
Run: uv run python scripts/run_copilot_model.py --serve
Ask: uv run python scripts/run_copilot_model.py --question 'Which graph bounds differ?' --language en
No GPU offload, remote endpoint, model training or paid API is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from akash_flood.copilot import FactStore, MODEL, ask_local_model


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--model", type=Path, default=ROOT / "data/copilot-model" / MODEL["file"])
    parser.add_argument("--proof", type=Path, default=ROOT / "outputs/access-bounds-proof/proof.json")
    parser.add_argument("--question")
    parser.add_argument("--language", choices=["en", "ne"], default="en")
    parser.add_argument("--port", type=int, default=8782)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise SystemExit("Use an unprivileged local port.")
    if args.serve:
        binary = shutil.which("llama-server")
        if binary is None:
            raise SystemExit("Install the official llama.cpp runtime first; no template fallback is used.")
        if not args.model.is_file() or args.model.stat().st_size != MODEL["bytes"]:
            raise SystemExit("The pinned official GGUF is absent or has the wrong size.")
        with args.model.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != MODEL["sha256"]:
            raise SystemExit("Pinned model checksum mismatch.")
        command = [binary, "-m", str(args.model), "--device", "none", "-ngl", "0",
                   "--no-op-offload", "-t", "4", "-tb", "4",
                   "-c", "4096", "-np", "1", "--host", "127.0.0.1", "--port", str(args.port),
                   "--alias", MODEL["repository"], "--cors-origins", "localhost", "--no-webui"]
        print(json.dumps({"command": command, "model": MODEL, "cpu_only": True}), flush=True)
        raise SystemExit(subprocess.call(command))
    if not args.question:
        parser.error("Choose --serve or provide --question.")
    store = FactStore.from_path(args.proof)
    print(json.dumps(ask_local_model(store, args.question, language=args.language,
                                    endpoint=f"http://127.0.0.1:{args.port}"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
