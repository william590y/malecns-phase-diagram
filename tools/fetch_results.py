#!/usr/bin/env python3
"""Download the large result arrays (.npz/.npy, about 61 GB in total) from Google Drive into results/.

They are stored as one uncompressed tar per directory group; drive_manifest.json lists each archive's Drive file id,
size, file count and sha256. Archives unpack into results/, next to the JSON/Markdown summaries already in the repo.

  python tools/fetch_results.py --list                 # show the archives and sizes
  python tools/fetch_results.py toplevel al_lit        # fetch and unpack the named archives
  python tools/fetch_results.py --all                  # everything (61 GB)

Needs `pip install gdown`. If Drive answers "too many users have viewed or downloaded this file", wait and retry, or
open the folder link in drive_manifest.json in a browser, or use rclone with your own Google account.
"""
import argparse, hashlib, json, os, subprocess, sys, tarfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
man = json.load(open(os.path.join(HERE, "drive_manifest.json")))
ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="*")
ap.add_argument("--all", action="store_true")
ap.add_argument("--list", action="store_true")
ap.add_argument("--keep-tar", action="store_true")
a = ap.parse_args()
arch = man["archives"]
if a.list or not (a.names or a.all):
    print("folder:", man["folder_url"])
    for k, v in arch.items():
        print(f"{k:28s} {v['tar_bytes'] / 1e9:7.2f} GB  {v['n_files']:5d} files  -> results/{v['unpacks_to']}")
    sys.exit(0)
tmp = os.path.join(HERE, "data", "_downloads"); os.makedirs(tmp, exist_ok=True)
for name in (list(arch) if a.all else a.names):
    v = arch[name]; tar = os.path.join(tmp, v["file"])
    subprocess.run([sys.executable, "-m", "gdown", "--continue", f"https://drive.google.com/uc?id={v['id']}", "-O", tar], check=True)
    h = hashlib.sha256()
    with open(tar, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    if h.hexdigest() != v["sha256"]: sys.exit(f"sha256 mismatch for {name}; delete {tar} and retry")
    with tarfile.open(tar) as t: t.extractall(os.path.join(HERE, "results"), filter="data")
    if not a.keep_tar: os.remove(tar)
    print("unpacked", name)
