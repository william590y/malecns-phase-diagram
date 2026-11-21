#!/usr/bin/env python3
"""Download the large result arrays (.npz/.npy, about 61 GB in total) into results/.

They are attached to the GitHub release "results-v1" of this repository: one uncompressed tar per directory group,
split into parts of at most 1.9 GB. results_manifest.json lists each archive's parts, size, file count and sha256.
Archives unpack into results/, next to the JSON/Markdown summaries already in the repo. No account is needed.

  python tools/fetch_results.py --list                 # show the archives and sizes
  python tools/fetch_results.py toplevel module        # fetch and unpack the named archives
  python tools/fetch_results.py --all                  # everything (61 GB)
"""
import argparse, hashlib, json, os, shutil, sys, tarfile, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
man = json.load(open(os.path.join(HERE, "results_manifest.json")))
ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="*")
ap.add_argument("--all", action="store_true")
ap.add_argument("--list", action="store_true")
ap.add_argument("--keep-tar", action="store_true")
a = ap.parse_args()
arch = man["archives"]
if a.list or not (a.names or a.all):
    print("release:", man["release_url"])
    for k, v in arch.items():
        print(f"{k:22s} {v['tar_bytes'] / 1e9:7.2f} GB  {v['n_files']:5d} files  {len(v['parts']):2d} part(s)")
    sys.exit(0)
tmp = os.path.join(HERE, "data", "_downloads"); os.makedirs(tmp, exist_ok=True)
for name in (list(arch) if a.all else a.names):
    v = arch[name]; tar = os.path.join(tmp, name + ".tar"); h = hashlib.sha256()
    with open(tar, "wb") as out:
        for part in v["parts"]:
            print("downloading", part, flush=True)
            with urllib.request.urlopen(man["download_base"] + part) as r:
                for b in iter(lambda: r.read(1 << 24), b""):
                    out.write(b); h.update(b)
    if h.hexdigest() != v["sha256"]: sys.exit(f"sha256 mismatch for {name}; delete {tar} and retry")
    with tarfile.open(tar) as t: t.extractall(os.path.join(HERE, "results"), filter="data")
    if not a.keep_tar: os.remove(tar)
    print("unpacked", name)
