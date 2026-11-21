# Instructions for agents

Read this before running anything. `README.md` says what the project found; `PROJECT_NOTES.md` has the model
definitions, the code map and the traps; `STATE.md` is the chronological log.

## 1. Environment
- Python 3.11+ with the packages in `requirements.txt`. The work used Python 3.13, torch 2.11 (CUDA 12.8).
- The simulators need one GPU with under 1 GB of free memory. Spectral scripts (`spectrum*.py`, `al_spectra.py`)
  are CPU-only scipy.
- `*.sbatch` files are Slurm runners written for one university cluster. Partition names, `--constraint` and
  `--exclude` lines will not match your cluster: edit them, or run the `python` line inside each file directly.

## 2. Get the connectome (public, not in this repository)
Source: MaleCNS v1.0 flat connectome, <https://male-cns.janelia.org/download/> (files under
`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`). No account is needed.

```bash
cd data_pipeline
python -m malecns.data download   --raw-dir ../data/raw                       # about 1.1 GB, checksummed
python -m malecns.data preprocess --raw-dir ../data/raw --output ../data/cache   # 399 MB of .npy arrays
python -m malecns.data inspect ../data/cache                                  # expect 167,216 neurons, 25,587,572 edges
cd ..
```
`data_pipeline/docs/data.md` documents every array, the inclusion rule and the neurotransmitter-to-sign map.
`data/` is git-ignored. Treat the cache as read-only.

## 3. Set the paths
The scripts use absolute paths. In this repository they appear as the placeholders `/MALECNS_ROOT`,
`/MALECNS_CACHE`, `/MALECNS_RAW` and `/MALECNS_PYBIN`. Run once:

```bash
python tools/set_paths.py                 # defaults: repo root, data/cache, data/raw, this python
python tools/set_paths.py --check         # should print "files still unset: 0"
```
This rewrites files in `code/` in place. Do not commit the rewritten paths; before committing code changes, restore
the placeholders (replace your absolute paths back) or commit only the lines you changed.

## 4. Get the raw result arrays (only if you need them)
The repository holds every report, table and JSON summary. The raw arrays (61 GB) are assets of the GitHub release
`results-v1` (https://github.com/william590y/malecns-phase-diagram/releases/tag/results-v1): one tar per directory group, split into parts of at most 1.9 GB (`<name>.tar.partNN`).
`results_manifest.json` lists the parts, size and sha256 of each archive. Downloads are public; no login is needed.

```bash
python tools/fetch_results.py --list
python tools/fetch_results.py toplevel module      # small ones first
python tools/fetch_results.py m2__decay            # one Model 2 lane (10 GB)
```
- The script downloads the parts, joins them, checks the sha256 and unpacks into `results/` at the original paths, so
  analysis scripts find the files. Archive names follow the directories: `m2__decay` is `results/m2/decay/`.
- By hand: download the parts, `cat name.tar.part* > name.tar`, then `tar -xf name.tar -C results/`.
- Do not add `.npz`/`.npy` files to git (`.gitignore` blocks them under `results/`).

## 5. Check the setup
```bash
python code/swc.py --exp decay --norm raw --g 0.2 0.3 0.4 --r 1 --T 500 --B 8 --out results/try_decay.npz
cd data_pipeline && python -m pytest tests -q && cd ..
```
Simulator tests: `code/m2_sim_test.py`, `code/m2_obs_test.py`, `code/ald_test.py`. With all new options off,
`code/ald_sim.py` must reproduce `code/swc.py` bit for bit (`code/ald_validate.py`).

## 6. Rules that matter
- Do not modify `code/swc.py`; later code is validated against it.
- For Model 2 depression always pass `--std_mode spike`. The default `rate` mode shifts the onset.
- The main normalisation is `--norm raw`.
- Slurm: never call `squeue`/`sacct` from inside array tasks or loops.
- Reports named `REPORT.md` were independently re-verified; files named `*_preverify.md` were not.
- `refs/` (a copyrighted paper PDF) and `logs/` are not published; paths to them in the notes will not resolve.
