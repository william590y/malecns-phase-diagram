"""E4 (weight transforms) figures from results/aldef/weights/summary.json -> figs/aldef/weights_*.png
  weights_spectrum.png  leading AL mode vs best non-AL mode per transform (equal mean in-strength units) + g_lin
  weights_onsets.png    S / on / D per transform at r = 0, 1 in two unit systems (equal mean in-strength; equal lambda_1)
  weights_curves.png    lifetime tau(g) and participation ratio PR(g) vs g / g_lin(T), all transforms, r = 0 and 1
  weights_ignition.png  composition of the activity at the first sustained point + carrier of the slow-window activity
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/MALECNS_ROOT"
S = json.load(open(f"{ROOT}/results/aldef/weights/summary.json"))
FIG = f"{ROOT}/figs/aldef"
TR = ["linear", "cap50", "cap20", "cap10", "sqrt", "log1p", "binary"]
LAB = {"linear": "linear", "cap50": "cap 50", "cap20": "cap 20", "cap10": "cap 10", "sqrt": "sqrt", "log1p": "log1p",
       "binary": "binary"}
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]       # fixed categorical order
CT = dict(zip(TR, COL))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white",
                     "legend.frameon": False})
x = np.arange(len(TR))


def run(n, r):
    return S["runs"].get(f"{n}_r{r}")


# ------------------------------------------------------------------------------------------------- spectrum
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
a = ax[0]
lamAL = [S["spec"][n]["first_AL_mode"]["lam"] for n in TR]
lamNA = [S["spec"][n]["first_nonAL_mode"]["lam"] for n in TR]
regNA = [S["spec"][n]["first_nonAL_mode"]["lead_region"] for n in TR]
w = 0.36
a.bar(x - w / 2 - 0.01, lamAL, w, color="#2a78d6", label="best AL mode", edgecolor="white", linewidth=1)
a.bar(x + w / 2 + 0.01, lamNA, w, color="#a9a8a3", label="best non-AL mode", edgecolor="white", linewidth=1)
for i in range(len(TR)):
    a.text(x[i] + w / 2, lamNA[i] + 0.08, regNA[i].replace("CB other", "CB").replace(" ", "\n"), ha="center", va="bottom",
           fontsize=7.5, color=INK2)
    a.text(x[i] - w / 2, lamAL[i] + 0.08, f"{lamAL[i]:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
a.set_xticks(x, [LAB[n] for n in TR]); a.set_ylabel(r"leading eigenvalue $\lambda$ (mean in-strength = 1)")
a.set_title("Excitatory operator: the AL keeps the top mode except under binary weights", fontsize=10, loc="left")
a.legend(loc="upper right"); a.grid(axis="x", visible=False)
a = ax[1]
glin = [S["spec"][n]["g_lin"] for n in TR]
a.plot(x, glin, "o-", color=INK2, lw=1.5, ms=6)
for i, n in enumerate(TR):
    a.text(x[i], glin[i] * 1.06, f"{glin[i]:.3f}", ha="center", fontsize=7.5, color=INK)
a.set_xticks(x, [LAB[n] for n in TR], rotation=30); a.set_ylabel(r"$g_{lin}$ (silence unstable)")
a.set_title(r"Linear threshold $g_{lin}$ (equal mean in-strength)", fontsize=10, loc="left")
a.set_ylim(0, max(glin) * 1.2)
fig.tight_layout(); fig.savefig(f"{FIG}/weights_spectrum.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------------------------------------- onsets
fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True)
MK = {"S": ("o", "none", r"S: $\tau$ = 200 (slow window starts)"), "on": ("o", "fill", r"on: $\tau$ = 3000 (sustained)"),
      "D": ("^", "fill", "D: PR = N/4 (delocalized)")}
for ri, r in enumerate((0, 1)):
    for ui, (ukey, ulab) in enumerate((("g", "g  (equal mean in-strength)"),
                                       ("g_lam", r"g $\lambda_1(T)/\lambda_1$(linear)  (equal leading eigenvalue)"))):
        a = ax[ri, ui]
        for i, n in enumerate(TR):
            R = run(n, r)
            if R is None:
                continue
            fac = 1.0 if ukey == "g" else S["spec"][n]["lam1"] / S["spec"]["linear"]["lam1"]
            for k, (m, fill, lab) in MK.items():
                v = R["bounds"].get(k)
                if v is None:
                    continue
                ci = R["bounds"].get(k + "_ci")
                if ci:
                    a.plot([x[i], x[i]], [ci[0] * fac, ci[1] * fac], color=CT[n], lw=1.2, alpha=0.7)
                a.plot(x[i], v * fac, m, ms=7, mfc=CT[n] if fill == "fill" else "white", mec=CT[n], mew=1.6)
            if R["bounds"].get("D") is None and R.get("D_not_reached_up_to"):
                a.annotate("", (x[i], R["D_not_reached_up_to"] * fac * 1.6), (x[i], R["D_not_reached_up_to"] * fac),
                           arrowprops=dict(arrowstyle="->", color=CT[n]))
            g0 = S["spec"][n]["g_lin"] * fac
            a.plot([x[i] - 0.28, x[i] + 0.28], [g0, g0], color=INK2, lw=1.4)
        a.set_yscale("log"); a.set_ylabel(ulab, fontsize=8.5)
        a.set_title(f"r = {r}" + ("  (no inhibition)" if r == 0 else "  (inhibition on)"), fontsize=10, loc="left")
        a.set_xticks(x, [LAB[n] for n in TR])
        lo, hi = a.get_ylim(); a.set_ylim(lo, hi * 1.5)
h = [plt.Line2D([], [], ls="", marker=m, ms=7, mfc=INK2 if f == "fill" else "white", mec=INK2, mew=1.6)
     for m, f, _ in MK.values()] + [plt.Line2D([], [], color=INK2, lw=1.4)]
l = [lab for _, _, lab in MK.values()] + [r"$g_{lin}$ (silence linearly unstable)"]
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Phase boundaries under weight transforms (fully active start, h = 0, T = 3000; bars = 95% CI)",
             fontsize=10.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{FIG}/weights_onsets.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------------------------------------- curves
fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.4))
for ri, r in enumerate((0, 1)):
    for i, n in enumerate(TR):
        R = run(n, r)
        if R is None:
            continue
        g = np.array(R["g"]) / S["spec"][n]["g_lin"]
        tau = np.array(R["tau_reg_curve"]); dd = np.array(R["deaths"])
        ok = dd > 0
        ax[0, ri].plot(g[ok], tau[ok], "-", color=CT[n], lw=1.6, label=LAB[n])
        last = np.where(ok)[0].max() if ok.any() else None
        if last is not None and last + 1 < len(g):           # first point where no replica died: arrow off the top
            ax[0, ri].plot(g[last + 1], 2.4e4, "^", color=CT[n], ms=6)
        on = R["bounds"]["on"]
        sus = np.array(R["g"]) >= (on or np.inf) * (1 - 1e-9)
        ax[1, ri].plot(g[sus], np.array(R["pr_curve"])[sus] * 167216, "-", color=CT[n], lw=1.6, label=LAB[n])
        if sus.any():
            ax[1, ri].plot(g[sus][0], np.array(R["pr_curve"])[sus][0] * 167216, "o", color=CT[n], ms=5)
    a = ax[0, ri]
    a.axhline(200, color=INK2, lw=0.9, ls="--"); a.axhline(3000, color=INK2, lw=0.9, ls="--")
    a.text(0.72, 240, r"$\tau$ = 200 (S)", fontsize=7.5, color=INK2); a.text(0.72, 3600, r"$\tau$ = 3000 (on)", fontsize=7.5, color=INK2)
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(8, 4e4); a.set_xlim(0.7, 3.2)
    a.set_xticks([0.7, 1, 1.5, 2, 3], ["0.7", "1", "1.5", "2", "3"]); a.minorticks_off()
    a.axvline(1, color=INK2, lw=0.8)
    a.set_title(f"r = {r}: mean lifetime from a fully active start", fontsize=10, loc="left")
    a.set_ylabel(r"$\tau$ (censored-exp MLE)"); a.set_xlabel(r"$g / g_{lin}(T)$")
    a = ax[1, ri]
    a.axhline(167216 / 4, color=INK2, lw=0.9, ls="--"); a.text(1.05, 167216 / 4 * 1.15, "N/4 (D)", fontsize=7.5, color=INK2)
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(500, 3e5); a.set_xlim(0.9, 250)
    a.set_xlabel(r"$g / g_{lin}(T)$"); a.set_ylabel("participation ratio (neurons)")
    a.set_title(f"r = {r}: localization of sustained activity (dot = onset)", fontsize=10, loc="left")
ax[0, 0].legend(loc="lower right", fontsize=8, ncol=2)
fig.text(0.01, 0.005, "▲ at the top = first g where no replica died within T = 3000.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(f"{FIG}/weights_curves.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------------------------------------- ignition site
GROUPS = [("AL lLN1", ["AL lLN1"]), ("AL lLN2", ["AL lLN2"]), ("AL PNs", ["AL uPN", "AL mPN"]),
          ("AL other", ["AL v/il3LN", "AL LN other", "AL lLN other", "AL IN/ON", "AL ORN", "AL HRN/TRN"]),
          ("MB KC", ["MB KC"]), ("MB other", ["MB MBON/DAN/APL/DPM"]), ("LH", ["LH"]), ("CB other", ["CB other"]),
          ("VNC", ["VNC"])]
GCOL = ["#1f5fae", "#4f8fe0", "#9cc0ee", "#cfe0f6", "#6250d6", "#b2a9ec", "#1baf7a", "#eda100", "#eb6834", "#c8c7c2"]


def full_cat(occ_desc):
    """Category shares (the summary keeps the top 10 categories; the remainder goes to 'other')."""
    cat = occ_desc["cat"]
    out = [sum(cat.get(c, 0) for c in cs) for _, cs in GROUPS]
    out.append(max(0.0, 1 - sum(out)))
    return out


fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
for ri, r in enumerate((0, 1)):
    a = ax[ri]
    for i, n in enumerate(TR):
        R = run(n, r)
        if R is None or "ignition" not in R:
            continue
        sh = full_cat(R["ignition"])
        left = 0
        for j, v in enumerate(sh):
            a.barh(i, v, left=left, color=GCOL[j], edgecolor="white", linewidth=1.2, height=0.7,
                   label=(GROUPS[j][0] if j < len(GROUPS) else "other") if i == 0 else None)
            left += v
        ig = R["ignition"]
        a.text(1.02, i - 0.17, f"{R.get('ignition_class', '')}", va="center", fontsize=8, color=INK, weight="bold")
        a.text(1.02, i + 0.2, f"g={ig['g']:.3g}, PR {ig['PR_neurons']:.0f}", va="center", fontsize=7.5, color=INK2)
    a.set_yticks(range(len(TR)), [LAB[n] for n in TR])
    if ri == 0:
        a.invert_yaxis()                                     # shared y: invert once
    a.set_xlim(0, 1.45); a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    a.set_xlabel("share of activity (occupancy) at the first sustained g"); a.grid(axis="y", visible=False)
    a.set_title(f"r = {r}: where the first sustained activity sits", fontsize=10, loc="left")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=10, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{FIG}/weights_ignition.png", dpi=150); plt.close(fig)
# ------------------------------------------------------------------------------------------------- causal AL tests
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
for ri, r in enumerate((0, 1)):
    a = ax[ri]
    a.axhspan(1 / 1.1, 1.1, color="#efeee9", zorder=0)
    a.axhline(1, color=INK2, lw=0.9)
    for kind, dx, mk, lab in (("noAL", -0.13, "v", "AL module clamped off"), ("ALonly", 0.13, "o", "AL module alone")):
        for i, n in enumerate(TR):
            c = S["causal"].get(f"{kind}_{n}_r{r}")
            if c is None or c["on_ratio"] is None:
                continue
            on0 = c["on_intact"]; ci = c["bounds"].get("on_ci")
            if ci:
                a.plot([x[i] + dx] * 2, [ci[0] / on0, ci[1] / on0], color=CT[n], lw=1.3)
            a.plot(x[i] + dx, c["on_ratio"], mk, ms=8, mfc=CT[n] if kind == "noAL" else "white", mec=CT[n], mew=1.7)
    a.set_yscale("log"); a.set_yticks([0.9, 1, 1.25, 1.5, 2, 3, 4], ["0.9", "1", "1.25", "1.5", "2", "3", "4"])
    a.minorticks_off()
    a.set_xticks(x, [LAB[n] for n in TR])
    a.set_title(f"r = {r}: sustained onset relative to the intact network", fontsize=10, loc="left")
    if ri == 0:
        a.set_ylabel("onset / intact onset")
        a.legend([plt.Line2D([], [], ls="", marker="v", ms=8, mfc=INK2, mec=INK2),
                  plt.Line2D([], [], ls="", marker="o", ms=8, mfc="white", mec=INK2, mew=1.7)],
                 ["AL module clamped off", "AL module alone"], loc="upper center", fontsize=8)
    a.text(-0.42, 1.055, "±10%", fontsize=7.5, color=INK2, va="center", ha="left")
fig.text(0.01, 0.01, "Filled ▼ far above 1 = the AL was needed; open ○ near 1 = the AL alone suffices; both above 1 = "
         "cooperative ignition; ▼ at 1 and ○ far above = the AL plays no part.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(f"{FIG}/weights_causal.png", dpi=150); plt.close(fig)
print("figures written")
