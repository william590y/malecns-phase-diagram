"""Figures for the AL data grounding (figs/al/). Reads results/al_lit/grounding/*.json (+ the cache for the weight CCDF)."""
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from al_common import load, labels, OUT, ROOT

FIG = f"{ROOT}/figs/al"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
COL = {"AL": "#2a78d6", "VNC": "#eb6834", "CX": "#1baf7a", "LH": "#eda100", "MB": "#e87ba4", "OL intrinsic": "#008300",
       "CB other": "#6250d6", "AN/DN": "#e34948"}
GRAY = "#a3a29c"
col = lambda r: COL.get(r, GRAY)
mod = json.load(open(f"{OUT}/module.json")); E = json.load(open(f"{OUT}/edges.json"))
tail = json.load(open(f"{OUT}/tail20.json")); rec = json.load(open(f"{OUT}/recruit.json"))
spec = {f.split("spec_")[1][:-5]: json.load(open(f)) for f in glob.glob(f"{OUT}/spec_*.json") if "peel" not in f}

# ---- fig 1: module composition and leading-mode weight ----------------------------------------------------------
cats = ["AL lLN1", "AL lLN2", "AL v/il3LN", "AL LN other", "AL uPN", "AL mPN", "AL IN/ON", "AL HRN/TRN", "AL ORN", "LH",
        "MB MBON/DAN/APL/DPM", "CB other", "AN/DN"]
bc = mod["by_category"]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
y = np.arange(len(cats))
ax[0].barh(y, [bc.get(c, {}).get("n_module", 0) for c in cats], color=COL["AL"], height=0.6)
for i, c in enumerate(cats):
    r = bc.get(c, {})
    ax[0].text(r.get("n_module", 0) + 3, i, f"{r.get('n_module', 0)} / {r.get('n_cns', 0)}", va="center", fontsize=8, color="#52514e")
ax[0].set_yticks(y, [c.replace("AL ", "") + (" (AL)" if c.startswith("AL") else "") for c in cats]); ax[0].invert_yaxis()
ax[0].set_xlabel("neurons in the module (label: module / whole CNS)"); ax[0].set_xlim(0, 300)
ax[0].set_title("What the module is made of", loc="left", fontsize=10)
h = 0.38
ax[1].barh(y - h / 2, [bc.get(c, {}).get("right1", 0) for c in cats], height=h, color=COL["AL"], label="right eigenvector (where activity sits)")
ax[1].barh(y + h / 2, [bc.get(c, {}).get("left1", 0) for c in cats], height=h, color=COL["VNC"], label="left eigenvector (who drives the mode)")
ax[1].set_xlabel("share of |v|² of the leading mode (λ₁ = 5.57)"); ax[1].set_xlim(0, 0.75)
ax[1].set_title("Who carries the leading mode", loc="left", fontsize=10); ax[1].legend(frameon=False, fontsize=8, loc="lower right")
ax[1].text(0.535, 0, "lLN1_bc (30 cells)\nalone: 51%", va="center", fontsize=8, color="#52514e")
fig.tight_layout(); fig.savefig(f"{FIG}/al_module_composition.png", dpi=160); plt.close(fig)

# ---- fig 2: robustness of the AL's dominance ------------------------------------------------------------------
order = ["full", "no_ORN", "no_AL_sensory", "soma_only", "ct_nt", "drop_ORN>PN", "drop_ORN>LN", "drop_ORN>ORN",
         "drop_PN>PN", "drop_PN>LN", "drop_LN>PN", "drop_PN<>LN", "drop_LN>LN", "cap50", "soma_cap50", "noORN_cap50",
         "cap20", "cap10", "sqrt", "log1p", "binary", "LN_nonexc", "no_AL"]
nice = {"full": "full operator", "no_ORN": "ORNs removed", "no_AL_sensory": "ORNs+HRNs+TRNs removed",
        "soma_only": "neurons with soma in volume only", "ct_nt": "cell-type-level NT signs", "cap50": "weights capped at 50",
        "soma_cap50": "soma only + cap 50", "noORN_cap50": "no ORNs + cap 50", "cap20": "weights capped at 20",
        "cap10": "weights capped at 10", "sqrt": "sqrt(weights)", "log1p": "log(1+w)", "binary": "binary (unweighted)",
        "LN_nonexc": "AL local neurons non-excitatory", "no_AL": "AL removed"}
fig, ax = plt.subplots(figsize=(9.4, 6.6))
for i, v in enumerate(order):
    s = spec[v.replace("/", "-")]; m = s["modes"][0]
    lead = max(m["region"], key=m["region"].get)
    ax.barh(i, s["lam"][0], color=col(lead), height=0.62)
    b = s.get("best_nonAL")
    txt = f"{s['lam'][0]:.2f}  ({lead} {m['region'][lead]*100:.0f}%)"
    if b and b["j"] > 0:
        ax.plot([b["lam"]], [i], marker="D", ms=5, color="#0b0b0b", mec="#fcfcfb", mew=1.2, zorder=3)
        txt += f"   ×{s['lam'][0] / b['lam']:.2f} over best non-AL"
    elif lead != "AL":
        al = [s["lam"][j] for j, a in enumerate(s["al_share_top"]) if a >= 0.5]
        txt += f"   (AL mode: {al[0]:.2f})" if al else "   (no AL mode in top 8)"
    ax.text(s["lam"][0] * 1.06, i, txt, va="center", fontsize=7.5, color="#0b0b0b")
ax.set_yticks(range(len(order)), [nice.get(v, v.replace("drop_", "drop ").replace(">", "→").replace("<→", "↔")) for v in order])
ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlim(0.15, 60)
ax.set_xlabel("leading eigenvalue λ₁ of the whole-CNS excitatory operator (raw units, c = 1/742.7; log scale)")
ax.set_title("Leading eigenvalue under perturbations. Bar colour = region holding the leading mode\n"
             "(blue AL, orange VNC, pink MB); black diamond = best mode not on the AL", loc="left", fontsize=9.5)
for yy in [4.5, 12.5, 20.5]:
    ax.axhline(yy, color="#e6e5e0", lw=0.8)
fig.tight_layout(); fig.savefig(f"{FIG}/al_robustness.png", dpi=160); plt.close(fig)

# ---- fig 3: why — weight statistics and region Perron roots -----------------------------------------------------
d = load(); L = labels()
pre, post, w = d["pre"], d["post"], d["weight"]
reg, cat = L["reg"], L["cat"]
core = (reg == "AL") & ~np.isin(cat, ["AL ORN", "AL HRN/TRN"])
sets = {"AL core (no sensory)": (core[pre] & core[post], COL["AL"]),
        "MB (KC, MBON, DAN, APL, DPM)": ((reg == "MB")[pre] & (reg == "MB")[post], COL["MB"]),
        "CX": ((reg == "CX")[pre] & (reg == "CX")[post], COL["CX"]),
        "OL intrinsic (columns)": ((reg == "OL intrinsic")[pre] & (reg == "OL intrinsic")[post], COL["OL intrinsic"]),
        "VNC": ((reg == "VNC")[pre] & (reg == "VNC")[post], COL["VNC"]),
        "central brain, other": ((reg == "CB other")[pre] & (reg == "CB other")[post], COL["CB other"])}
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
xs = np.unique(np.round(np.logspace(0, 3.3, 60)))
for name, (m, cc) in sets.items():
    ww = np.sort(w[m]); cc_ = 1 - np.searchsorted(ww, xs, side="left") / len(ww)
    ax[0].plot(xs, cc_, color=cc, lw=2, label=name)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(1e-5, 1.1)
ax[0].set_xlabel("synapses per connection, s"); ax[0].set_ylabel("fraction of connections with ≥ s synapses")
ax[0].set_title("Connection strength inside each region", loc="left", fontsize=10); ax[0].legend(frameon=False, fontsize=7.5)
R = E["regions"]
rn = ["AL", "VNC", "CB other", "CX", "OL intrinsic", "LH", "MB", "VPN/VCN"]
x = np.arange(len(rn)); hw = 0.27
for j, (key, lab, a) in enumerate([("perron", "raw synapse counts", 1.0), ("perron_cap50", "capped at 50", 0.6),
                                   ("perron_binary", "binary (×10)", 0.3)]):
    vals = [R[r][key]["lam"] * (10 if key == "perron_binary" else 1) for r in rn]
    ax[1].bar(x + (j - 1) * hw, vals, width=hw - 0.03, color=[col(r) for r in rn], alpha=a, label=lab)
ax[1].set_xticks(x, [r.replace(" intrinsic", "") for r in rn], rotation=30, ha="right")
ax[1].set_ylabel("Perron root of the region's own excitatory subgraph")
ax[1].set_title("Each region alone: raw / cap 50 / binary×10", loc="left", fontsize=10)
ax[1].legend(frameon=False, fontsize=7.5, handlelength=1)
lg = ax[1].get_legend()
for hnd in lg.legend_handles:
    hnd.set_color("#52514e")
sub = E["al_subgraphs"]
names = ["whole AL", "AL LNs + PNs", "all AL LNs", "lLN1 + lLN2", "lLN1_bc only", "AL minus lLN1_bc", "AL minus lLN1, lLN2",
         "AL minus all LNs", "AL ORNs only", "AL PNs only"]
vals = [sub[n]["lam"] for n in names]
ax[2].barh(range(len(names)), vals, color=[COL["AL"] if v > 1.9 else GRAY for v in vals], height=0.6)
for i, v in enumerate(vals):
    ax[2].text(v + 0.06, i, f"{v:.2f}", va="center", fontsize=8)
ax[2].axvline(1.879, color="#52514e", lw=1, ls=(0, (3, 3)))
ax[2].set_yticks(range(len(names)), [n + (" (30 cells)" if n == "lLN1_bc only" else "") for n in names]); ax[2].invert_yaxis()
ax[2].set_xlabel("Perron root of the AL subgraph"); ax[2].set_xlim(0, 6.6)
ax[2].set_title("Inside the AL: the cholinergic LN core", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/al_why.png", dpi=160); plt.close(fig)

# ---- fig 4: exact decomposition of lambda --------------------------------------------------------------------
dec = E["decomposition"]["by_group"][:8]
fig, ax = plt.subplots(figsize=(7, 3.2))
lab = [r["pair"].replace(" > ", " → ") for r in dec]
ax.barh(range(len(dec)), [r["share"] for r in dec], color=COL["AL"], height=0.6)
for i, r in enumerate(dec):
    ax.text(r["share"] + 0.01, i, f"{r['share']*100:.1f}%   ({r['exc_synapses']/1e3:.0f}k exc. synapses)", va="center", fontsize=8)
ax.set_yticks(range(len(dec)), lab); ax.invert_yaxis(); ax.set_xlim(0, 0.95)
ax.set_xlabel("share of λ₁ = Σ u_i M_ij v_j / (u·v) carried by each edge class")
ax.set_title("LN→LN loops carry 65% of λ₁ (lLN1_bc→lLN1_bc alone 42%); ORN edges < 1%", loc="left", fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{FIG}/al_lambda_decomposition.png", dpi=160); plt.close(fig)

# ---- fig 5: the spectral tail and the simulated switch-on order ------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
modes = tail["modes"]
regs = ["AL", "VNC", "AN/DN", "CB other", "CX", "LH", "MB"]
bottom = np.zeros(len(modes))
for r in regs + ["other"]:
    v = np.array([m["region"].get(r, 0) if r != "other" else 1 - sum(m["region"].get(q, 0) for q in regs) for m in modes])
    ax[0].bar(np.arange(1, 21), v, bottom=bottom, color=col(r) if r != "other" else GRAY, width=0.8, label=r,
              edgecolor="#fcfcfb", linewidth=0.8)
    bottom += v
for m in modes:
    ax[0].text(m["k"], 1.02, f"{m['lam']:.2f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
ax[0].set_ylim(0, 1.22); ax[0].set_xticks(range(1, 21)); ax[0].set_xlabel("mode k (by Re λ; λ above each bar)")
ax[0].set_ylabel("share of |v_k|² by region")
ax[0].set_title("Top-20 excitatory modes: AL (1, 2, 4, 6), then VNC/GNG motor, CX ring at 9–10", loc="left", fontsize=9.5)
ax[0].legend(frameon=False, fontsize=7, ncol=4, loc="upper right", bbox_to_anchor=(1, 0.97))
r0 = rec["0.0"]; g = np.array(r0["g"])
show = {"mode1": ("1 AL (lLN1_bc core)", COL["AL"]), "mode4": ("4 AL (lLN2F/P)", "#7fb1ea"),
        "mode3": ("3 VNC leg premotor/MN", COL["VNC"]), "mode7": ("7 GNG motor (MN11D)", COL["CB other"]),
        "mode9": ("9 VNC + CX (ExR6/4)", COL["CX"]), "mode5": ("5 VNC abdominal", "#c98500"),
        "mode17": ("17 DNg33/AN", COL["AN/DN"])}
for k, (lab, cc) in show.items():
    ax[1].plot(g, r0["mode_occ"][k], color=cc, lw=2, label=lab)
ax[1].set_xscale("log"); ax[1].set_xlim(0.1, 3)
ax[1].set_xlabel("coupling g  (r = 0, h = 0, fully active start, T = 3000)")
ax[1].set_ylabel("mean occupancy on the mode's 90% support")
ax[1].set_title("Simulated switch-on order matches the spectral order", loc="left", fontsize=10)
ax[1].legend(frameon=False, fontsize=7.5, loc="upper left")
for k in ["mode1", "mode3"]:
    gk = 0.906 / tail["modes"][int(k[4:]) - 1]["lam"]
    ax[1].axvline(gk, color=show[k][1], lw=1, ls=(0, (3, 3)))
    ax[1].text(gk * 1.02, 0.02 if k == "mode1" else 0.26, f"α_eff/λ_{k[4:]} = {gk:.3f}", fontsize=7, color=show[k][1], rotation=90, va="bottom")
fig.tight_layout(); fig.savefig(f"{FIG}/al_tail_order.png", dpi=160); plt.close(fig)
print("figures written to", FIG)
