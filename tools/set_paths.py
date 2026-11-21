#!/usr/bin/env python3
"""Point the code at your machine. The scripts in code/ use absolute paths, published here as placeholders:

  /MALECNS_ROOT   this repository's root (code/, results/, figs/, logs/ live under it)
  /MALECNS_CACHE  the preprocessed MaleCNS v1.0 cache (see AGENTS.md, "Get the connectome")
  /MALECNS_RAW    the raw Janelia downloads (the .feather files)
  /MALECNS_PYBIN  the directory that holds the python interpreter to use

Run once after cloning:   python tools/set_paths.py --cache /abs/path/data/cache --raw /abs/path/data/raw
It rewrites files in code/ only, in place. Run with --check to list what is still unset.
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=HERE)
ap.add_argument("--cache", default=os.path.join(HERE, "data", "cache"))
ap.add_argument("--raw", default=os.path.join(HERE, "data", "raw"))
ap.add_argument("--pybin", default=os.path.dirname(sys.executable))
ap.add_argument("--check", action="store_true")
a = ap.parse_args()
pairs = [("/MALECNS_ROOT", os.path.abspath(a.root)), ("/MALECNS_CACHE", os.path.abspath(a.cache)),
         ("/MALECNS_RAW", os.path.abspath(a.raw)), ("/MALECNS_PYBIN", os.path.abspath(a.pybin))]
n = 0
for f in sorted(os.listdir(os.path.join(HERE, "code"))):
    p = os.path.join(HERE, "code", f)
    if not os.path.isfile(p): continue
    try: s = open(p, encoding="utf-8").read()
    except UnicodeDecodeError: continue
    if a.check:
        if "/MALECNS_" in s: print("unset:", f); n += 1
        continue
    t = s
    for k, v in pairs: t = t.replace(k, v)
    if t != s: open(p, "w", encoding="utf-8").write(t); n += 1
print(("files still unset: " if a.check else "files rewritten: ") + str(n))
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
