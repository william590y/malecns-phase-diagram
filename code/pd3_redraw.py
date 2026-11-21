"""pd3 redraw_fss part A: the (g, r) phase diagram redrawn as a continuous heat map of the measured module lifetime (CPU).

No new simulation. Inputs = exactly analysis2_phase.py's simulation files and pooling (load_sim copied verbatim below; T_c = 3000:
runs with T = 8000 / 30000 at r = 1 are truncated at the last recorded time <= 3000), and the verified boundaries / theory lines
of results/analysis2/phase_complete.json.
Per (graph, r) and g: tau_sim = E/d (censored-exponential MLE); 0 deaths -> censored, lower bound 2E/chi2_.95(2) stored.
  PR/N from occ (T = 3000 runs only).
Per r row: ln tau_reg (= E/max(d, 0.5), analysis2's regularisation) made monotone in g by weighted isotonic regression
  (weights d + 0.5), interpolated linearly in ln g onto 320 ln-uniform points in [0.1, 30]; cells outside the row's simulated
  g range are blank (hatched "not simulated"). No interpolation across r: each r is a band between the midpoints of its
  neighbours.
Panel A: log10 tau on [0, log10 3000] (cividis); cells with tau_iso >= 3000 in a flat light colour ("sustained over T = 3000").
Panel B: log10(PR/N) (isotonic PR among the points at/above the tau = 3000 crossing, as analysis2's D criterion) in the
  sustained region, viridis on [-3, 0]; other simulated cells grey.
Lines: g_lin solid (the only solid line); dashed = verified simulated tau = 200, tau = 3000, PR = N/4 (phase_complete.json sim,
  thin band = 'total' 95% uncertainty), inline label "(crossover)"; dotted lighter = theory P2 tau = 200 / 3000 and QMF PR = N/4.
Panel C: drift of the r = 1 onset with observation time T: measured g_half (longruns.md) and the K = 36.9 chain (theory lambda(g)
  from the dense branch rows, recursion of theory_branch_gpu.py) as a continuous curve, with theory_bd.json real_crossings.
Consistency check: matplotlib contour of the interpolated ln tau field (x = ln g) at ln 200 / ln 3000, read off at each row,
  against phase_complete.json; max |d ln g| must be <= one grid step (ln(300)/319 = 0.0179).
Outputs: results/pd3/redraw/heatmap_{real,null,wnull}.npz, heatmap_meta.json; figs/pd3/phase_lifetime_heatmap.png,
  phase_lifetime_heatmap_nulls.png.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd3_common import iso, first_cross, log_tau_smooth, A_EFF, ROOT

RES = f"{ROOT}/results"; OUT = f"{RES}/pd3/redraw"; FIG = f"{ROOT}/figs/pd3"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
N = 167216; T_REF = 3000.0; TAU_S = 200.0; PR_D = 0.25
NG = 320; GMIN, GMAX = 0.1, 30.0
GRID = np.exp(np.linspace(np.log(GMIN), np.log(GMAX), NG))
STEP = float(np.log(GMAX / GMIN) / (NG - 1))
EDGES = np.exp(np.r_[np.log(GRID[0]) - STEP / 2, 0.5 * (np.log(GRID[1:]) + np.log(GRID[:-1])), np.log(GRID[-1]) + STEP / 2])
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "verm": "#D55E00", "purple": "#CC79A7", "black": "#000000"}
LINECOL = {"S": OI["verm"], "on": OI["blue"], "D": OI["purple"]}
LINELAB = {"S": "τ = 200", "on": "τ = 3000 (onset)", "D": "PR = N/4"}
SUSTAIN_COL = "#EFE6D2"          # flat light colour for tau >= T
NOSIM_COL = "#D9D9D9"
PJ = json.load(open(f"{RES}/analysis2/phase_complete.json"))


# ------------------------------------------------------- copied verbatim from analysis2_phase.py (file list + load_sim)
def run_kind(m):
    if not m.get("null"):
        return "real"
    return "wnull" if m.get("null_kind") == "weights" else "null"


SIM_FILES = (sorted(glob.glob(f"{RES}/pd/decay_r*_*.npz")) + sorted(glob.glob(f"{RES}/pd2/onset_*_r*.npz")) +
             sorted(glob.glob(f"{RES}/pd2/deloc_*_r*.npz")) +
             [f"{RES}/{x}.npz" for x in ["decay_raw_r1", "decay_raw_r1_null", "fine_decay_raw_r1", "fine_decay_raw_r1_null",
                                         "upper_decay_raw_r1"]] + [f"{RES}/pd/long_real.npz", f"{RES}/pd/long_null.npz"])


def load_sim():
    pts = {}                                   # (kind, r) -> list of point dicts
    used = []
    for p in SIM_FILES:
        if not os.path.exists(p):
            continue
        d = np.load(p)
        m = json.loads(str(d["meta"]))
        ok = (m.get("exp") == "decay" and m.get("norm") == "raw" and float(m.get("h", 0)) == 0.0 and
              abs(float(m.get("dt", 0.2)) - 0.2) < 1e-9 and float(m.get("beta", 1)) == 1.0 and float(m["T"]) >= T_REF - 1e-6)
        if not ok:
            continue
        t, g, surv = d["t"], d["g"], d["surv"]
        win = t <= T_REF + 1e-6
        Tc = float(t[win].max())
        occ = d["occ"].astype(np.float64) if (abs(float(m["T"]) - T_REF) < 1e-6 and "occ" in d.files) else None
        kind, r = run_kind(m), float(m["r"])
        lab = os.path.relpath(p, RES)
        used.append({"file": lab, "kind": kind, "r": r, "T": float(m["T"]), "B": int(m["B"]), "n_g": int(len(g)),
                     "g_range": [float(g.min()), float(g.max())], "T_c": Tc})
        for i, gg in enumerate(g):
            s = surv[i][win]                          # [nt, B]
            dead = s == 0
            died = dead.any(0)
            ext = np.where(died, t[win][np.argmax(dead, 0)], Tc)
            pr = None
            if occ is not None:
                o = occ[i]
                pr = float(o.sum() ** 2 / (o ** 2).sum() / N) if o.sum() > 0 else 0.0
            pts.setdefault((kind, r), []).append({"g": float(gg), "times": ext.astype(float), "died": died.astype(bool),
                                                  "pr": pr, "B": int(s.shape[1]), "src": lab})
    merged = {}
    for key, lst in pts.items():
        lst.sort(key=lambda x: x["g"])
        out = []
        for x in lst:
            if out and abs(x["g"] / out[-1]["g"] - 1) < 1e-4:
                y = out[-1]
                y["times"] = np.r_[y["times"], x["times"]]; y["died"] = np.r_[y["died"], x["died"]]
                if x["pr"] is not None:
                    y["prs"].append((x["pr"], x["B"]))
                y["B"] += x["B"]; y["src"] += "+" + x["src"]
            else:
                out.append({"g": x["g"], "times": x["times"], "died": x["died"], "B": x["B"], "src": x["src"],
                            "prs": [(x["pr"], x["B"])] if x["pr"] is not None else []})
        for y in out:
            y["pr"] = (sum(a * b for a, b in y["prs"]) / sum(b for _, b in y["prs"])) if y["prs"] else None
            d_, E = int(y["died"].sum()), float(y["times"].sum())
            y["deaths"], y["E"] = d_, E
            y["tau"] = E / d_ if d_ > 0 else float("inf")
            y["tau_reg"] = E / max(d_, 0.5)
            if d_ > 0:
                y["tau_ci"] = [2 * E / stats.chi2.ppf(0.975, 2 * d_ + 2), 2 * E / stats.chi2.ppf(0.025, 2 * d_)]
            else:
                y["tau_ci"] = [2 * E / stats.chi2.ppf(0.95, 2), float("inf")]
            y["alive"] = float(1 - y["died"].mean())
        merged[key] = out
    return merged, used


# ---------------------------------------------------------------------------------------------------- per-row fields
def row_field(P):
    g = np.array([p["g"] for p in P]); d = np.array([p["deaths"] for p in P], float); E = np.array([p["E"] for p in P])
    lt = iso(np.log(E / np.maximum(d, 0.5)), d + 0.5)                     # isotonic ln tau_reg
    inside = (GRID >= g.min() * (1 - 1e-9)) & (GRID <= g.max() * (1 + 1e-9))
    ltg = np.full(NG, np.nan); ltg[inside] = np.interp(np.log(GRID[inside]), np.log(g), lt)
    # censored-cell bookkeeping: nearest data point (in ln g) has 0 deaths
    near = np.array([int(np.argmin(np.abs(np.log(g) - np.log(x)))) for x in GRID])
    cens_near = (d[near] == 0) & inside
    # PR among points at/above the tau = 3000 crossing (analysis2 sim_boundaries D rule)
    on = first_cross(g, lt, np.log(T_REF))
    prg = np.full(NG, np.nan); pr_range = None
    if on["status"] in ("ok", "below_grid"):
        k0 = on.get("k", 0)
        sel = [j for j in range(k0, len(P)) if P[j]["pr"] is not None]
        if sel:
            gs = g[sel]; prs = iso([P[j]["pr"] for j in sel], np.ones(len(sel)))
            m = (GRID >= gs.min() * (1 - 1e-9)) & (GRID <= gs.max() * (1 + 1e-9))
            prg[m] = np.interp(np.log(GRID[m]), np.log(gs), np.log10(np.maximum(prs, 1e-12)))
            pr_range = [float(gs.min()), float(gs.max())]
    S = first_cross(g, lt, np.log(TAU_S)); half_ = None
    return {"lt": ltg, "inside": inside, "cens_near": cens_near, "log10pr": prg, "g_range": [float(g.min()), float(g.max())],
            "pr_range": pr_range, "S_direct": S["g"], "on_direct": on["g"], "n_points": len(P),
            "n_censored_points": int((d == 0).sum())}


def r_edges(rs):
    rs = np.asarray(rs, float)
    mid = 0.5 * (rs[1:] + rs[:-1])
    lo = rs[0] - (mid[0] - rs[0]) if len(rs) > 1 else rs[0] - 0.5
    hi = rs[-1] + (rs[-1] - mid[-1]) if len(rs) > 1 else rs[-1] + 0.5
    return np.r_[lo, mid, hi]


def contour_check(kind, rs, F):
    """matplotlib contour of the ln tau field (x = ln g, y = r rows) at ln 200 / ln 3000; crossing read at each row y = r."""
    fig, ax = plt.subplots()
    Z = np.ma.masked_invalid(np.array([F[r]["lt"] for r in rs]))
    cs = ax.contour(np.log(GRID), np.array(rs), Z, levels=[np.log(TAU_S), np.log(T_REF)])
    res = {}
    for lev, key in zip(cs.allsegs, ["S", "on"]):
        verts = np.vstack([s for s in lev if len(s)]) if any(len(s) for s in lev) else np.zeros((0, 2))
        rows = []
        for r in rs:
            on_row = verts[np.abs(verts[:, 1] - r) < 1e-9] if len(verts) else np.zeros((0, 2))
            gc = float(np.exp(on_row[:, 0].min())) if len(on_row) else None     # first (lowest-g) crossing on that row
            ent = PJ["sim"][kind][f"{r:g}"][key]
            gv = ent["g"] if ent["status"] == "ok" else None
            dl = abs(np.log(gc / gv)) if (gc and gv) else None
            tot = ent.get("total")
            ins = bool(gc is not None and tot and tot[0] is not None and tot[1] is not None and tot[0] <= gc <= tot[1])
            rows.append({"r": r, "contour_g": gc, "verified_g": gv, "abs_dlng": dl, "n_crossings_on_row": int(len(on_row)),
                         "verified_total_95": tot, "contour_inside_total_band": ins,
                         "direct_first_cross": F[r]["S_direct" if key == "S" else "on_direct"]})
        dls = [x["abs_dlng"] for x in rows if x["abs_dlng"] is not None]
        res[key] = {"rows": rows, "max_abs_dlng": float(max(dls)) if dls else None, "n_compared": len(dls),
                    "n_inside_total_band": int(sum(x["contour_inside_total_band"] for x in rows)),
                    "n_missing": int(sum(1 for x in rows if x["verified_g"] is not None and x["contour_g"] is None))}
    plt.close(fig)
    return res


# ------------------------------------------------------------------------------------------------------ chain T-drift
def chain_drift():
    parts = sorted(glob.glob(f"{RES}/pd/branch_real_dense_part*.json"))
    rows = sorted([x for p in parts for x in json.load(open(p))["rows"]], key=lambda x: x["g"])
    G = np.array([x["g"] for x in rows]); lam = np.array([x["lam"] for x in rows])
    gf = np.exp(np.linspace(np.log(G[0]), np.log(0.6), 500))
    lt = np.array([log_tau_smooth(float(np.interp(np.log(x), np.log(G), lam)), 36.9) for x in gf])
    Ts = np.geomspace(100, 1e5, 60)
    gT = [first_cross(gf, lt, np.log(T))["g"] for T in Ts]
    bd = json.load(open(f"{RES}/pd3/theory_bd.json"))["real_crossings"]["K_needed_r1_36.9"]
    return {"T": Ts.tolist(), "g": gT, "K": 36.9, "lam_source": "results/pd/branch_real_dense_part*.json (lam), K = 36.9",
            "theory_bd_on_tau3000": bd["on_tau3000"], "theory_bd_on_tau30000": bd["on_tau30000"],
            "ours_at_3000": first_cross(gf, lt, np.log(3000.0))["g"], "ours_at_30000": first_cross(gf, lt, np.log(30000.0))["g"],
            "first_dense_row_g": float(G[0])}


# ------------------------------------------------------------------------------------------------------------ drawing
def stroke(lw=3.0):
    return [pe.withStroke(linewidth=lw, foreground="white")]


def draw_lines(ax, kind, rs_sim, label_lines=True, theory=True, panel="A"):
    glin = PJ["g_lin"][kind]
    ax.axvline(glin, color="k", lw=2.0, ls="-", zorder=6)
    for key in ["S", "on", "D"]:
        rr, gv, lo, hi = [], [], [], []
        for r in rs_sim:
            e = PJ["sim"][kind][f"{r:g}"][key]
            if e["status"] in ("ok", "at_onset") and e["g"]:
                rr.append(r); gv.append(e["g"])
                tot = e.get("total") or [e["g"], e["g"]]
                lo.append(tot[0] if tot[0] else e["g"]); hi.append(tot[1] if tot[1] else e["g"])
        if rr:
            ax.fill_betweenx(rr, lo, hi, color=LINECOL[key], alpha=0.28, lw=0, zorder=5)
            ax.plot(gv, rr, ls=(0, (5, 2.5)), color=LINECOL[key], lw=2.0, zorder=7, marker="o", ms=2.8,
                    path_effects=stroke(3.6))
            if label_lines:
                at_on = key == "D" and all(PJ["sim"][kind][f"{r:g}"]["D"]["status"] == "at_onset" for r in rr)
                frac = {"S": 0.80, "on": 0.40, "D": 0.22 if at_on else 0.5}[key]
                k = int(np.argmin(np.abs(np.array(rr) - (rr[0] + frac * (rr[-1] - rr[0])))))
                txt = {"S": "τ = 200\n(crossover)", "on": "τ = 3000\n(crossover)",
                       "D": "PR = N/4\n(at the onset)" if at_on else "PR = N/4\n(crossover)"}[key]
                left = key == "S" or at_on
                ax.annotate(txt, (gv[k], rr[k]), xytext=(-9 if left else 9, 0), textcoords="offset points", fontsize=8.5,
                            color=LINECOL[key], ha="right" if left else "left", va="center", zorder=9, fontweight="bold",
                            path_effects=stroke(2.8))
        if theory:
            th = PJ["theory"][kind]
            tr, tg = [], []
            for rk, t in sorted(th.items(), key=lambda kv: float(kv[0])):
                v = t.get(key)
                if isinstance(v, dict) and v.get("g"):
                    tr.append(float(rk)); tg.append(v["g"])
            if tr:
                ax.plot(tg, tr, ls=(0, (1, 1.6)), color=LINECOL[key], lw=1.6, alpha=0.85, zorder=6, path_effects=stroke(2.6))


def heat_panels(axA, axB, kind, rs, F, title_kind, label_lines=True, show_cbar=True, cax=None, caxB=None):
    re = r_edges(rs)
    for ax in (axA, axB):
        ax.add_patch(Rectangle((GMIN, re[0]), GMAX - GMIN, re[-1] - re[0], facecolor=NOSIM_COL, edgecolor="#9a9a9a",
                               hatch="////", lw=0, zorder=0))
    LT = np.array([F[r]["lt"] for r in rs])
    L10 = LT / np.log(10)
    sust = np.isfinite(LT) & (LT >= np.log(T_REF))
    A = np.ma.masked_where(~np.isfinite(L10) | sust, np.clip(L10, 0, np.log10(T_REF)))
    m1 = axA.pcolormesh(EDGES, re, A, cmap="cividis", norm=Normalize(0, np.log10(T_REF)), zorder=1, shading="flat")
    axA.pcolormesh(EDGES, re, np.ma.masked_where(~sust, np.ones_like(LT)), cmap=ListedColormap([SUSTAIN_COL]), zorder=1,
                   shading="flat")
    PRm = np.array([F[r]["log10pr"] for r in rs])
    nons = np.isfinite(LT) & ~sust
    axB.pcolormesh(EDGES, re, np.ma.masked_where(~nons, np.ones_like(LT)), cmap=ListedColormap(["#BDBDBD"]), zorder=1,
                   shading="flat")
    Bv = np.ma.masked_where(~(sust & np.isfinite(PRm)), PRm)
    m2 = axB.pcolormesh(EDGES, re, Bv, cmap="viridis", norm=Normalize(-3, 0), zorder=1, shading="flat")
    top = re[-1] + 0.13 * (re[-1] - re[0])           # white label strip above the data
    for ax in (axA, axB):
        ax.set_xscale("log"); ax.set_xlim(GMIN, GMAX); ax.set_ylim(re[0], top)
        ax.axhline(re[-1], color="#777777", lw=0.8, zorder=2)
        gl = PJ["g_lin"][kind]
        ax.text(gl * 1.06, 0.5 * (re[-1] + top), f"g_lin = {gl:.4g}: mean-field critical line\n(silence loses stability; r-independent)",
                fontsize=8.3, ha="left", va="center", zorder=10)
        ax.set_xlabel("coupling g (raw-normalised units)", fontsize=10)
        ax.set_yticks(rs if len(rs) <= 8 else [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6])
        ax.tick_params(labelsize=9)
        draw_lines(ax, kind, rs, label_lines=label_lines)
    axA.set_ylabel("inhibition scale r", fontsize=10)
    return m1, m2


def main():
    sim, used = load_sim()
    meta = {"inputs": {"simulation_files": used, "boundaries_and_theory": "results/analysis2/phase_complete.json"},
            "grid": {"n": NG, "g_min": GMIN, "g_max": GMAX, "step_ln_g": STEP}, "consistency_check": {}, "rows": {}}
    Fall = {}
    for kind in ["real", "null", "wnull"]:
        rs = sorted(r for (k, r) in sim if k == kind)
        F = {r: row_field(sim[(kind, r)]) for r in rs}
        Fall[kind] = (rs, F)
        LT = np.array([F[r]["lt"] for r in rs])
        np.savez_compressed(f"{OUT}/heatmap_{kind}.npz", g=GRID, g_edges=EDGES, r=np.array(rs), r_edges=r_edges(rs),
                            ln_tau_iso=LT, log10_tau=LT / np.log(10),
                            sustained=np.isfinite(LT) & (LT >= np.log(T_REF)),
                            censored_nearest=np.array([F[r]["cens_near"] for r in rs]),
                            inside=np.array([F[r]["inside"] for r in rs]),
                            log10_PR_over_N=np.array([F[r]["log10pr"] for r in rs]), g_lin=PJ["g_lin"][kind])
        meta["consistency_check"][kind] = contour_check(kind, rs, F)
        sus = np.isfinite(LT) & (LT >= np.log(T_REF))
        cn = np.array([F[r]["cens_near"] for r in rs])
        meta["rows"][kind] = {f"{r:g}": {"g_range": F[r]["g_range"], "pr_range": F[r]["pr_range"], "n_points": F[r]["n_points"],
                                         "n_censored_points": F[r]["n_censored_points"]} for r in rs}
        meta[f"sustained_cells_{kind}"] = {"n_sustained": int(sus.sum()),
                                           "frac_of_sustained_whose_nearest_point_has_0_deaths": float(cn[sus].mean()) if sus.any() else None,
                                           "n_cells_censored_nearest_but_below_T": int((cn & ~sus & np.isfinite(LT)).sum())}
        c = meta["consistency_check"][kind]
        print(kind, "rows", len(rs), "contour check max |dln g|: S", c["S"]["max_abs_dlng"], "on", c["on"]["max_abs_dlng"],
              "missing", c["S"]["n_missing"], c["on"]["n_missing"], flush=True)
    drift = chain_drift(); meta["chain_drift_r1"] = drift
    meta["longruns_g_half_r1"] = {"T3000": 0.3508, "T8000": 0.3659, "T30000": [0.3862, 0.399],
                                  "source": "results/analysis2/longruns.md (T-drift table, corrections item 5)"}
    meta["g_lin"] = PJ["g_lin"]
    json.dump(meta, open(f"{OUT}/heatmap_meta.json", "w"), indent=1, default=float)

    # ------------------------------------------------------------------------------------------ figure: real network
    rs, F = Fall["real"]
    fig = plt.figure(figsize=(16.5, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.62], height_ratios=[1, 0.055], wspace=0.22, hspace=0.32,
                          left=0.05, right=0.985, top=0.9, bottom=0.08)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1], sharey=axA)
    m1, m2 = heat_panels(axA, axB, "real", rs, F, "MaleCNS")
    axA.set_title("A  Measured module lifetime τ_sim (decay from fully active, T = 3000)", fontsize=11, loc="left")
    axB.set_title("B  Spatial extent of sustained activity, PR/N", fontsize=11, loc="left")
    cA = fig.add_subplot(gs[1, 0]); cB = fig.add_subplot(gs[1, 1])
    cb = fig.colorbar(m1, cax=cA, orientation="horizontal"); cb.set_label("log10 τ_sim (model time 1/α)", fontsize=9.5)
    cb.ax.tick_params(labelsize=9)
    cb2 = fig.colorbar(m2, cax=cB, orientation="horizontal"); cb2.set_label("log10 (participation ratio / N)", fontsize=9.5)
    cb2.ax.tick_params(labelsize=9)
    gl = PJ["g_lin"]["real"]
    # region labels
    axA.text(0.105, 4.0, "dies out\n(τ < 200)", fontsize=9, color="white", ha="left", va="center", zorder=10)
    axB.text(29, 2.6, "PR < N/4 up to g = 30 for r = 1.75, 2, 3, 4, 6\n(r = 2.25–5.5 otherwise simulated only to g = 2.6)",
             fontsize=7.8, ha="right", va="center", zorder=10, bbox=dict(facecolor="white", alpha=0.85, lw=0))
    axA.text(4.0, 3.2, "sustained over T = 3000\n(τ ≥ 3000)", fontsize=9, color="#333333", ha="center", va="center", zorder=10)
    # Panel C: T drift
    axC = fig.add_subplot(gs[0, 2])
    axC.plot(drift["T"], drift["g"], color=OI["blue"], lw=2, label="chain, K = 36.9 (theory λ(g)): τ(g) = T")
    axC.plot([3000, 30000], [drift["theory_bd_on_tau3000"], drift["theory_bd_on_tau30000"]], "o", mfc="white",
             mec=OI["blue"], ms=8, mew=1.8, label="theory_bd.json crossings 0.346 / 0.395", zorder=5)
    axC.errorbar([3000, 8000], [0.3508, 0.3659], yerr=[[0.3508 - 0.344, 0.3659 - 0.3585], [0.36 - 0.3508, 0.3728 - 0.3659]],
                 fmt="s", color=OI["verm"], ms=7, capsize=3, label="simulated g_half (95% CI)", zorder=6)
    axC.errorbar([30000], [0.5 * (0.3862 + 0.399)], yerr=[[0.5 * (0.399 - 0.3862)], [0.5 * (0.399 - 0.3862)]], fmt="s",
                 color=OI["verm"], mfc="white", ms=7, capsize=3, label="g_half T = 30000 (0.386–0.399, model-dependent)", zorder=6)
    axC.axhline(gl, color="k", lw=1.6)
    axC.text(130, gl - 0.003, "g_lin (T-independent)", fontsize=8.5, va="top")
    axC.set_xscale("log"); axC.set_xlim(100, 1e5); axC.set_ylim(0.15, 0.43)
    axC.set_xlabel("observation time T (model time)", fontsize=10); axC.set_ylabel("apparent onset g at r = 1", fontsize=10)
    axC.set_title("C  The 'onset' drifts with T (r = 1)", fontsize=11, loc="left")
    axC.legend(fontsize=7.6, loc="lower right", bbox_to_anchor=(1.0, 0.07), frameon=True)
    axC.tick_params(labelsize=9); axC.grid(alpha=0.25, lw=0.6)
    handles = [Line2D([], [], color="k", lw=2, label="g_lin (solid: the only critical line)"),
               Line2D([], [], color=LINECOL["S"], lw=2, ls=(0, (5, 2.5)), label="simulated τ = 200 (dashed, 95% band)"),
               Line2D([], [], color=LINECOL["on"], lw=2, ls=(0, (5, 2.5)), label="simulated τ = 3000"),
               Line2D([], [], color=LINECOL["D"], lw=2, ls=(0, (5, 2.5)), label="simulated PR = N/4"),
               Line2D([], [], color="#555555", lw=1.6, ls=(0, (1, 1.6)), label="dotted: theory (P2 τ lines, QMF PR = N/4)"),
               Patch(facecolor=SUSTAIN_COL, edgecolor="#999999", label="sustained over T = 3000 (τ ≥ 3000)"),
               Patch(facecolor=NOSIM_COL, edgecolor="#9a9a9a", hatch="////", label="not simulated"),
               Patch(facecolor="#BDBDBD", label="B: not sustained")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8.8, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.savefig(f"{FIG}/phase_lifetime_heatmap.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------------------------------------ figure: nulls
    fig = plt.figure(figsize=(14.5, 11.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.05], hspace=0.36, wspace=0.14, left=0.06, right=0.985, top=0.9,
                          bottom=0.06)
    names = {"null": "configuration null (wiring shuffled)", "wnull": "weight-shuffled null"}
    for i, kind in enumerate(["null", "wnull"]):
        rs, F = Fall[kind]
        axA = fig.add_subplot(gs[i, 0]); axB = fig.add_subplot(gs[i, 1], sharey=axA)
        m1, m2 = heat_panels(axA, axB, kind, rs, F, names[kind], label_lines=True)
        axA.set_title(f"{'AC'[i]}  {names[kind]}: lifetime τ_sim  (g_lin = {PJ['g_lin'][kind]:.4f})", fontsize=11, loc="left")
        axB.set_title(f"{'BD'[i]}  {names[kind]}: PR/N in the sustained region", fontsize=11, loc="left")
    cA = fig.add_subplot(gs[2, 0]); cB = fig.add_subplot(gs[2, 1])
    cb = fig.colorbar(m1, cax=cA, orientation="horizontal"); cb.set_label("log10 τ_sim (model time)", fontsize=9.5)
    cb2 = fig.colorbar(m2, cax=cB, orientation="horizontal"); cb2.set_label("log10 (PR / N)", fontsize=9.5)
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8.8, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.savefig(f"{FIG}/phase_lifetime_heatmap_nulls.png", dpi=150)
    plt.close(fig)
    print("saved figures", flush=True)


if __name__ == "__main__":
    main()
