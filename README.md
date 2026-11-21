# MaleCNS phase diagram

Code and results for a study of collective dynamics on the *Drosophila* male central nervous system connectome
(MaleCNS v1.0, Janelia: 167,216 neurons, 25.6 million connections). A stochastic Wilson–Cowan model runs on the real
wiring, one binary unit per neuron, and the question is what phases it has as the coupling `g` and the
inhibition strength `r` change, and whether the connectome's spectrum predicts them. The first target was a
Griffiths phase (Moretti & Muñoz 2013).

## Main results

**Model 1** (plain stochastic Wilson–Cowan; `code/swc.py`; report `results/analysis2/phase_complete.md`)
- The leading excitatory eigenmodes are localized on one module, the antennal lobe (AL). Degree-preserving random
  rewiring removes the localization.
- Activity first becomes long-lived on that module. This gives a slow window that looks Griffiths-like, but lifetimes
  in it are exponential, the signature of one rare region. It is not a Griffiths phase.
- A linear onset theory from the spectrum (`g_lin = 0.1632`) matches simulation at `r = 0` and overstates inhibition
  for `r > 0`.
- A de Candia-style diagram (`results/pd3/REPORT.md`) finds one continuous line, no left/right symmetry breaking and
  no first-order lines.

**Model 2** (Model 1 plus short-term synaptic depression and spontaneous olfactory drive; `code/ald_sim.py`;
spec `results/m2/MODEL2.md`; report `results/m2/REPORT.md`)
- The onset rises 3.4–4.8× and the first sustained state is brain-wide; the AL holds only 5–6% of it.
- No Griffiths-like window. The transition is continuous and rounded.
- No `g` passes all three comparisons with fly data at `r` 0 and 1.

Whether the AL result is a model defect is examined in `results/aldef/REPORT.md`; the related sleep, arousal and
anaesthesia literature is in `results/al_lit/REPORT.md`.

## What is here

| path | contents |
| --- | --- |
| `code/` | simulators (GPU, PyTorch), spectral and theory code, analyses, figure scripts, Slurm runners |
| `results/` | every report (`REPORT.md`), plan, table and JSON summary. Raw arrays are on Google Drive (below) |
| `figs/` | all figures and the activity movie |
| `explainer/` | an illustrated walk-through as web pages; open `explainer/index.html` in a browser |
| `data_pipeline/` | downloads the public MaleCNS release and builds the array cache the code reads |
| `tools/` | `set_paths.py` (point the code at your machine), `fetch_results.py` (get raw arrays from Drive) |
| `AGENTS.md` | step-by-step setup and data instructions, written for coding agents |
| `PROJECT_NOTES.md`, `STATE.md` | the working notes and the full log of the project, as they were |

## Data that is not in the repository

- **The connectome** is public (CC BY 4.0): <https://male-cns.janelia.org/download/>. `data_pipeline/` downloads it
  (about 1.1 GB) and builds a 399 MB cache. See `AGENTS.md`.
- **Raw simulation arrays** (`.npz`/`.npy`, about 61 GB) are in a Google Drive folder:
  __FOLDER_URL__ . `python tools/fetch_results.py --list` shows the archives; fetch only what you need.
  Every number in the reports can also be regenerated from the code.

## Quick start

```bash
pip install -r requirements.txt
cd data_pipeline && python -m malecns.data download --raw-dir ../data/raw \
  && python -m malecns.data preprocess --raw-dir ../data/raw --output ../data/cache && cd ..
python tools/set_paths.py            # fills in the placeholder paths in code/
python code/swc.py --exp decay --norm raw --g 0.2 0.3 0.4 --r 1 --T 500 --B 8 --out results/try_decay.npz
```

The connectome data is © Janelia FlyEM, CC BY 4.0; cite the MaleCNS release when using it.
