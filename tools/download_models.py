#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
import sys
# Allow both direct script execution and module-style imports in tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from urllib.parse import urlparse


def normalize_url(url: str) -> str:
    """Convert common browser URLs to direct-download URLs."""
    parsed = urlparse(url)
    if parsed.netloc == "github.com" and "/blob/" in parsed.path:
        # https://github.com/owner/repo/blob/branch/path ->
        # https://raw.githubusercontent.com/owner/repo/branch/path
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] == "blob":
            owner, repo, _, branch = parts[:4]
            rest = "/".join(parts[4:])
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rest}"
    return url
from model_manifest import load_manifest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_or_download(url: str, dst: Path, timeout: int) -> None:
    parsed = urlparse(url)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if parsed.scheme in ("", "file"):
        src = Path(parsed.path if parsed.scheme == "file" else url).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"local model source not found: {src}")
        shutil.copyfile(src, dst)
        return
    with urllib.request.urlopen(url, timeout=timeout) as resp, dst.open("wb") as f:
        shutil.copyfileobj(resp, f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download PointPillars model assets from manifest")
    ap.add_argument("--manifest", default="models/model_manifest.json", help="model manifest JSON path")
    ap.add_argument("--dry-run", action="store_true", help="print planned actions without downloading")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--timeout", type=int, default=60, help="network timeout seconds")
    ap.add_argument("--allow-example-url", action="store_true", help="allow example.com placeholder URLs")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    for asset in manifest["assets"]:
        name = asset["name"]
        original_url = asset["url"]
        url = normalize_url(original_url)
        if url != original_url:
            print(f"[{name}] normalized URL: {original_url} -> {url}")
        out = Path(asset["output"])
        expected = (asset.get("sha256") or "").strip().lower()
        if "example.com" in url and not args.allow_example_url:
            print(f"[{name}] placeholder URL detected: {url}", file=sys.stderr)
            print("Replace models/model_manifest.json with real URLs or pass --allow-example-url for dry-run only.", file=sys.stderr)
            if not args.dry_run:
                return 2
        print(f"[{name}] {url} -> {out}")
        if args.dry_run:
            continue
        if out.exists() and not args.force:
            if expected:
                actual = sha256_file(out)
                if actual != expected:
                    raise RuntimeError(f"sha256 mismatch for existing {out}: {actual} != {expected}; use --force")
            print(f"[{name}] exists, skip")
            continue
        with tempfile.NamedTemporaryFile(prefix=out.name + ".", dir=str(out.parent if out.parent.exists() else Path(".")), delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            copy_or_download(url, tmp_path, args.timeout)
            if expected:
                actual = sha256_file(tmp_path)
                if actual != expected:
                    raise RuntimeError(f"sha256 mismatch for {name}: {actual} != {expected}")
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.replace(out)
            print(f"[{name}] saved: {out} ({out.stat().st_size} bytes)")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
