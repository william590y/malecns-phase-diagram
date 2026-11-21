"""m2_decay_figs.py -- figures of lane m2/decay (figs/m2/decay_*.png). Reads results/m2/decay/{boundaries,points,
where,lifetimes,osc}.json (code/m2_decay_report.py), results/analysis2/phase_complete.json (M1) and, for the M1 order
parameters, the M1 raw runs listed in phase_complete.json (cached in results/m2/decay/m1_order.json).
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/m2/decay"
FIG = f"{ROOT}/figs/m2"
N = 167216
G_LIN = 0.1632
PH_COL = {"Q": "#d9dbe3", "S": "#e8a33d", "L": "#2a78b8", "D": "#7b2a8c"}
PH_LAB = {"Q": "quiescent (τ < 200)", "S": "slow window (200 ≤ τ < 3000)", "L": "localized sustained (PR < N/4)",
          "D": "delocalized sustained (PR ≥ N/4)"}
B_COL = {"S": "#2a6fdb", "on": "#e4572e", "D": "#17a589"}
B_LAB = {"S": "S starts (τ = 200)", "on": "sustained onset (τ = 3000)", "D": "delocalized (PR = N/4)"}
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def jl(name):
    return json.load(open(f"{OUT}/{name}"))


def num(x):
    if x is None:
        return np.nan
    if isinstance(x, str):
        return np.inf if x == "inf" else np.nan
    return float(x)


# ------------------------------------------------------------------------------------------------ M1 per-point data
def m1_points():
    pc = json.load(open(f"{ROOT}/results/analysis2/phase_complete.json"))
    out = {}
    for gk, gm in (("real", "real"), ("cnull", "null"), ("wnull", "wnull")):
        for rk, e in pc["sim"][gm].items():
            P = []
            for p in e["points"]:
                tau = num(p["tau_sim"]) if p["deaths"] > 0 else np.inf
                ph = "Q" if tau < 200 else "S" if tau < 3000 else ("D" if (p["PR"] is not None and p["PR"] >= 0.25) else "L")
                P.append({"g": p["g"], "phase": ph, "PRN": p["PR"]})
            out[(gk, float(rk))] = {"points": P, "S": e["S"], "on": e["on"], "D": e["D"]}
    return out


def m1_order():
    """QS active fraction among replicas alive at T (mean of the records with t > 1500) and AL-module share of the
    second-half occupancy, for every M1 point with T = 3000 and occ (analysis2 inputs). Cached."""
    cf = f"{OUT}/m1_order.json"
    if os.path.exists(cf):
        return {tuple(json.loads(k)): v for k, v in json.load(open(cf)).items()}
    pc = json.load(open(f"{ROOT}/results/analysis2/phase_complete.json"))
    mod = np.load(f"{ROOT}/results/module/module_graph.npz", allow_pickle=True)["idx"]
    res = {}
    for inp in pc["inputs"]["simulation"]:
        if abs(inp["T"] - 3000) > 1e-6:
            continue
        d = np.load(f"{ROOT}/results/{inp['file']}")
        if "occ" not in d.files:
            continue
        t = d["t"]; w = t > 1500
        for i, g in enumerate(d["g"]):
            o = d["occ"][i].astype(np.float64)
            al = d["surv"][i][-1] > 0
            qs = float(d["rho"][i][w][:, al].mean()) if al.any() else np.nan
            sh = float(o[mod].sum() / o.sum()) if o.sum() > 0 else np.nan
            k = (inp["kind"], float(inp["r"]))
            res.setdefault(k, []).append([float(g), qs, sh, int(al.sum())])
    json.dump({json.dumps(list(k)): sorted(v) for k, v in res.items()}, open(cf, "w"))
    return {k: sorted(v) for k, v in res.items()}


# ------------------------------------------------------------------------------------------------ tiles
def tile_row(ax, y, pts, h=0.84, gmin=0.04, gmax=35, osc=None):
    g = np.array([p["g"] for p in pts]); o = np.argsort(g); g = g[o]
    ph = [pts[i]["phase"] for i in o]
    lg = np.log(g)
    edges = np.r_[lg[0] - (lg[1] - lg[0]) / 2 if len(lg) > 1 else lg[0] - 0.05,
                  (lg[1:] + lg[:-1]) / 2, lg[-1] + (lg[-1] - lg[-2]) / 2 if len(lg) > 1 else lg[0] + 0.05]
    E = np.exp(edges)
    for i in range(len(g)):
        ax.add_patch(Rectangle((E[i], y - h / 2), E[i + 1] - E[i], h, fc=PH_COL[ph[i]], ec="none"))
    # not simulated
    if E[0] > gmin:
        ax.add_patch(Rectangle((gmin, y - h / 2), E[0] - gmin, h, fc="white", ec="#bbbbbb", hatch="////", lw=0))
    if E[-1] < gmax:
        ax.add_patch(Rectangle((E[-1], y - h / 2), gmax - E[-1], h, fc="white", ec="#bbbbbb", hatch="////", lw=0))
    if osc:
        for gg, strong in osc:
            ax.plot(gg, y, marker="*" if strong else "o", ms=6 if strong else 3.5, mfc="white", mec="black", mew=0.7)


def tiles_panel(ax, rows, title, gmin=0.045, gmax=35, oscmap=None, marks=None, rlist=None):
    rs = sorted(rlist if rlist is not None else rows)
    for j, r in enumerate(rs):
        if r not in rows:
            ax.add_patch(Rectangle((gmin, j - 0.42), gmax - gmin, 0.84, fc="white", ec="#bbbbbb", hatch="////", lw=0))
            continue
        tile_row(ax, j, rows[r], gmin=gmin, gmax=gmax, osc=(oscmap or {}).get(r))
        if marks and r in marks:
            for b, gg in marks[r].items():
                if gg is not None and np.isfinite(gg):
                    ax.plot([gg, gg], [j - 0.42, j + 0.42], color="black", lw=1.0)
    ax.set_xscale("log"); ax.set_xlim(gmin, gmax); ax.set_ylim(-0.6, len(rs) - 0.4)
    ax.set_yticks(range(len(rs))); ax.set_yticklabels([f"{r:g}" for r in rs])
    ax.axvline(G_LIN, color="black", ls=":", lw=0.9)
    ax.set_xlabel("coupling g"); ax.set_ylabel("inhibition ratio r"); ax.set_title(title)


def fig_tiles(P, B):
    m1 = m1_points()
    m1rows = {r: v["points"] for (gk, r), v in m1.items() if gk == "real"}
    m2rows, oscm, mk2, mk1 = {}, {}, {}, {}
    for k, pl in P.items():
        e = B["rows"][k]
        if e["model"] == "M2" and e["graph"] == "real":
            m2rows[e["r"]] = pl
            oscm[e["r"]] = [(p["g"], p["osc_strong"]["total"]) for p in pl if any(p["osc"].values())]
            mk2[e["r"]] = {b: num(e[b]["g"]) for b in ("S", "on", "D")}
    for (gk, r), v in m1.items():
        if gk == "real":
            mk1[r] = {b: num(v[b]["g"]) for b in ("S", "on", "D")}
    fig, axs = plt.subplots(1, 2, figsize=(13, 7.2), sharey=True)
    rl = sorted(set(m1rows) | set(m2rows))
    tiles_panel(axs[0], m1rows, "M1 (old model, analysis2): no depression", marks=mk1, rlist=rl)
    tiles_panel(axs[1], m2rows, "M2 (new model): per-spike depression on excitatory outputs", oscmap=oscm, marks=mk2, rlist=rl)
    axs[1].set_ylabel("")
    for ax in axs:
        ax.text(G_LIN * 1.04, len(m2rows) - 0.5, "g_lin", fontsize=7, va="bottom")
    h = [Patch(fc=PH_COL[k], label=PH_LAB[k]) for k in "QSLD"]
    h += [Patch(fc="white", ec="#bbbbbb", hatch="////", label="not simulated"),
          Line2D([], [], color="black", lw=1, label="interpolated boundary (S, onset, D)"),
          Line2D([], [], color="black", ls=":", lw=0.9, label="g_lin = 0.163 (same in both)"),
          Line2D([], [], ls="", marker="o", mfc="white", mec="black", label="oscillation flag (OSC; ★ = OSC-strong)")]
    axs[1].text(0.99, -0.085, "In M2 'localized' (PR < N/4) still means 13,000-42,000 neurons; in M1 it starts at ~1,000 (the AL module).",
                transform=axs[1].transAxes, ha="right", va="top", fontsize=7.5, style="italic")
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle("Decay from a fully active start, h = 0, T = 3000: phase of every simulated point (MaleCNS)", y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(f"{FIG}/decay_phase_tiles.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------------------ boundary lines
def lines_panel(ax, data, title, ghost=None, gmax=35):
    rs = sorted(data)
    yy = np.array(rs)
    G = {b: np.array([num(data[r][b]["g"]) for r in rs]) for b in ("S", "on", "D")}
    st = {b: [data[r][b]["status"] for r in rs] for b in ("S", "on", "D")}
    # region shading per r band
    mids = np.r_[yy[0] - 0.06, (yy[1:] + yy[:-1]) / 2, yy[-1] + 0.06]
    for j, r in enumerate(rs):
        lo, hi = mids[j], mids[j + 1]
        s, o, d = G["S"][j], G["on"][j], G["D"][j]
        if st["D"][j] == "at_onset":
            d = o
        edges = [0.01, s, o, d if np.isfinite(d) else gmax * 2, gmax * 2]
        for ph, a, b in zip("QSLD", edges[:-1], edges[1:]):
            if np.isfinite(a) and np.isfinite(b) and b > a:
                ax.fill_between([a, b], lo, hi, color=PH_COL[ph], alpha=0.35, lw=0)
    for b in ("S", "on", "D"):
        ok = np.isfinite(G[b])
        if b == "D":
            ok &= np.array([s_ == "ok" for s_ in st["D"]])
        lo = np.array([num((data[r][b]["total"] or [None, None])[0]) for r in rs])
        hi = np.array([num((data[r][b]["total"] or [None, None])[1]) for r in rs])
        xerr = np.vstack([G[b] - lo, hi - G[b]])
        ax.errorbar(G[b][ok], yy[ok], xerr=np.where(np.isfinite(xerr[:, ok]), xerr[:, ok], 0), fmt="o-" if b != "D" else "D-",
                    ms=4.5, color=B_COL[b], mec="black", mew=0.5, lw=1.2, capsize=1.5, elinewidth=0.8, label=B_LAB[b])
        if b == "D":
            ao = np.array([s_ == "at_onset" for s_ in st["D"]])
            if ao.any():
                ax.plot(G["on"][ao], yy[ao], "D", ms=9, mfc="none", mec=B_COL["D"], mew=1.5, label="delocalized already at the onset")
            nn = np.array([s_ in ("none", "no_sustained_points") for s_ in st["D"]])
            if nn.any():
                xl = np.array([num((data[r]["D"].get("bracket") or [None])[0]) for r in rs])
                xl = np.where(np.isfinite(xl), xl, gmax * 0.93)
                ax.plot(xl[nn], yy[nn], ">", ms=6, mfc="none", mec=B_COL["D"], mew=1.2,
                        label="not delocalized up to the largest simulated g (marker at that g)")
    if ghost:
        rs2 = sorted(ghost)
        for b in ("S", "on", "D"):
            gg = np.array([num(ghost[r][b]["g"]) for r in rs2])
            ok = np.isfinite(gg) & np.array([ghost[r][b]["status"] == "ok" for r in rs2])
            ax.plot(gg[ok], np.array(rs2)[ok], "--", color=B_COL[b], lw=0.9, alpha=0.55)
        ax.plot([], [], "--", color="grey", lw=0.9, label="M1 boundaries (for comparison)")
    ax.axvline(G_LIN, color="black", ls=":", lw=0.9)
    ax.set_xscale("log"); ax.set_xlim(0.1, gmax); ax.set_ylim(yy[0] - 0.15, yy[-1] + 0.15)
    ax.set_xlabel("coupling g  (raw synapse-count weights / mean in-strength)"); ax.set_ylabel("inhibition ratio r")
    ax.set_title(title)


def fig_lines(B):
    m1 = m1_points()
    d1 = {r: v for (gk, r), v in m1.items() if gk == "real"}
    d2 = {e["r"]: e for k, e in B["rows"].items() if e["model"] == "M2" and e["graph"] == "real"}
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True, sharex=True)
    lines_panel(axs[0], d1, "M1 (analysis2)")
    lines_panel(axs[1], d2, "M2 (dashed: M1)", ghost=d1)
    axs[1].set_ylabel("")
    for ax in axs:
        ax.text(G_LIN * 1.04, ax.get_ylim()[1] - 0.1, "g_lin = 0.163", fontsize=7, va="top")
    hs, ls_ = axs[1].get_legend_handles_labels()
    h1, l1 = axs[0].get_legend_handles_labels()
    for h, l in zip(h1, l1):
        if l not in ls_:
            hs.append(h); ls_.append(l)
    hs += [Patch(fc=PH_COL[k], alpha=0.35, label=PH_LAB[k]) for k in "QSLD"]
    ls_ += [PH_LAB[k] for k in "QSLD"]
    fig.legend(hs, ls_, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Phase boundaries (h = 0, fully active start, T = 3000); error bars = grid bracket ∪ 95% replica bootstrap",
                 y=0.99)
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    fig.savefig(f"{FIG}/decay_phase_lines.png", dpi=150); plt.close(fig)


def fig_nulls(P, B):
    m1 = m1_points()
    fig, axs = plt.subplots(2, 2, figsize=(12, 5.6), sharex=True)
    for i, (gk, lab) in enumerate((("cnull", "configuration null (targets permuted)"),
                                   ("wnull", "weight-shuffled null (counts shuffled)"))):
        r1 = {r: v["points"] for (g_, r), v in m1.items() if g_ == gk}
        r2, osc2 = {}, {}
        for k, pl in P.items():
            e = B["rows"][k]
            if e["model"] == "M2" and e["graph"] == gk:
                r2[e["r"]] = pl
                osc2[e["r"]] = [(p["g"], p["osc_strong"]["total"]) for p in pl if any(p["osc"].values())]
        mk1 = {r: {b: num(v[b]["g"]) for b in ("S", "on", "D")} for (g_, r), v in m1.items() if g_ == gk}
        mk2 = {e["r"]: {b: num(e[b]["g"]) for b in ("S", "on", "D")} for e in B["rows"].values()
               if e["model"] == "M2" and e["graph"] == gk}
        rl = sorted(set(r1) | set(r2))
        tiles_panel(axs[i, 0], r1, f"M1, {lab}", marks=mk1, rlist=rl)
        tiles_panel(axs[i, 1], r2, f"M2, {lab}", oscmap=osc2, marks=mk2, rlist=rl)
        for ax in axs[i]:
            ax.axvline({"cnull": 0.8287, "wnull": 0.663}[gk], color="#555555", ls="-.", lw=0.9)
    h = [Patch(fc=PH_COL[k], label=PH_LAB[k]) for k in "QSLD"]
    h += [Patch(fc="white", ec="#bbbbbb", hatch="////", label="not simulated"),
          Line2D([], [], color="#555555", ls="-.", lw=0.9, label="null's own g_lin (0.829 / 0.663)"),
          Line2D([], [], color="black", ls=":", lw=0.9, label="g_lin of the real graph"),
          Line2D([], [], color="black", lw=1, label="interpolated boundary"),
          Line2D([], [], ls="", marker="o", mfc="white", mec="black", label="oscillation flag (OSC)")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Null graphs: M1 (left) vs M2 (right); same null graphs (null seed 0)", y=0.99)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(f"{FIG}/decay_phase_nulls.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------------------ order parameters
def heat_panel(ax, rows, val, cmap, vmin, vmax, title, cb_label, fig, rlist=None):
    rs = sorted(rlist if rlist is not None else rows)
    for j, r in enumerate(rs):
        if r not in rows:
            continue
        pts = sorted(rows[r], key=lambda x: x[0])
        g = np.array([p[0] for p in pts]); v = np.array([p[1] for p in pts])
        lg = np.log(g)
        if len(lg) < 2:
            continue
        E = np.exp(np.r_[lg[0] - (lg[1] - lg[0]) / 2, (lg[1:] + lg[:-1]) / 2, lg[-1] + (lg[-1] - lg[-2]) / 2])
        cm = plt.get_cmap(cmap)
        for i in range(len(g)):
            if np.isfinite(v[i]):
                c = cm((np.clip(v[i], vmin, vmax) - vmin) / (vmax - vmin))
                ax.add_patch(Rectangle((E[i], j - 0.42), E[i + 1] - E[i], 0.84, fc=c, ec="none"))
    ax.set_xscale("log"); ax.set_xlim(0.1, 35); ax.set_ylim(-0.6, len(rs) - 0.4)
    ax.set_yticks(range(len(rs))); ax.set_yticklabels([f"{r:g}" for r in rs], fontsize=6.5)
    ax.set_title(title, fontsize=9); ax.set_xlabel("coupling g")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02).set_label(cb_label, fontsize=8)


def fig_order(P, B):
    mo = m1_order()
    m1 = m1_points()
    # M1: qs, AL share from raw files; PR from phase_complete points
    q1 = {r: [[g, np.log10(qs) if qs and qs > 0 else np.nan] for g, qs, sh, na in v] for (k, r), v in mo.items() if k == "real"}
    a1 = {r: [[g, sh] for g, qs, sh, na in v] for (k, r), v in mo.items() if k == "real"}
    p1 = {r: [[p["g"], p["PRN"] if p["phase"] in "LD" else np.nan] for p in v["points"] if p["PRN"] is not None]
          for (k, r), v in m1.items() if k == "real"}
    q2, a2, p2 = {}, {}, {}
    for k, pl in P.items():
        e = B["rows"][k]
        if e["model"] == "M2" and e["graph"] == "real":
            r = e["r"]
            q2[r] = [[p["g"], np.log10(num(p["qs_tot"])) if num(p["qs_tot"]) > 0 else np.nan] for p in pl]
            a2[r] = [[p["g"], num(p["AL_share"]) if p["alive_T"] > 0 else np.nan] for p in pl]
            p2[r] = [[p["g"], num(p["PRN"]) if p["phase"] in "LD" else np.nan] for p in pl]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True)
    rl = sorted(set(q1) | set(q2))
    for i, (q, p, a, lab) in enumerate(((q1, p1, a1, "M1"), (q2, p2, a2, "M2"))):
        heat_panel(axs[i, 0], q, None, "viridis", -4.5, -0.3, f"{lab}: QS active fraction among survivors (t > 1500)", "log10 fraction active", fig, rl)
        heat_panel(axs[i, 1], p, None, "viridis", 0, 0.9, f"{lab}: participation ratio / N (sustained points)", "PR / N", fig, rl)
        heat_panel(axs[i, 2], a, None, "viridis", 0, 0.55, f"{lab}: share of activity in the AL module (526 neurons)", "AL-module share", fig, rl)
        axs[i, 0].set_ylabel("inhibition ratio r")
    fig.suptitle("Order parameters at the end of the run (blank = all replicas silent, or not simulated). "
                 "M1 QS fraction: mean of the log-grid records t > 1500; M2: mean over all steps 1500-3000", y=0.995, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(f"{FIG}/decay_order.png", dpi=140); plt.close(fig)


# ------------------------------------------------------------------------------------------------ lifetimes
def fig_lifetimes(LT):
    per = LT["long"]["per_g"]; ver = LT["long"]["verdict"]
    keys = [k for k in ("M2_real_r0", "M2_real_r1", "M2_real_r4", "M2_cnull_r1") if k in per]
    if not keys:
        return
    fig, axs = plt.subplots(3, len(keys), figsize=(4.6 * len(keys), 12.6), squeeze=False,
                            gridspec_kw={"hspace": 0.62})
    cm = plt.get_cmap("plasma")
    for j, k in enumerate(keys):
        L = per[k]; n = max(len(L) - 1, 1)
        ax = axs[0, j]
        for i, x in enumerate(L):
            km = np.array(x["km"]); lc = x["post_hoc"]["late200"]
            ax.step(km[:, 0], km[:, 1], where="post", color=cm(i / n), lw=1.1,
                    label=f"g {x['g']:.3g}: {lc['early_frac']:.2f}, {num(lc.get('tau_late')):.0f}, {num(lc.get('KS_p')):.2f}")
            tl = num(lc.get("tau_late"))
            if np.isfinite(tl) and lc["n_alive_tc"] >= 8:
                tt = np.linspace(200, min(30000, 200 + 8 * tl), 50)
                ax.plot(tt, (1 - lc["early_frac"]) * np.exp(-(tt - 200) / tl), ls="--", color=cm(i / n), lw=0.7)
        ax.set_yscale("log"); ax.set_ylim(5e-3, 1.05); ax.set_xscale("symlog", linthresh=100); ax.set_xlim(0, 30000)
        ax.set_xlabel("t (model units; linear below 100, log above)"); ax.set_ylabel("fraction of replicas alive (Kaplan–Meier)")
        ax.set_title(f"{k.replace('_', ' ')}\npre-registered (4.2): {ver[k]['verdict']}", fontsize=8.5)
        ax.legend(fontsize=6.3, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2,
                  title="g: fraction dead by t = 200, τ_late, KS p of the residual lifetimes (post hoc);\n"
                        "dashed = exponential fitted after t = 200", title_fontsize=6.3)
        ax = axs[1, j]
        for i, x in enumerate(L):
            h = np.array([q for q in x["post_hoc"]["hazard"] if q[4] >= 2])
            if len(h):
                ax.errorbar(h[:, 0] * (1 + 0.03 * i), h[:, 1], yerr=[h[:, 1] - h[:, 2], h[:, 3] - h[:, 1]], fmt="o-", ms=3,
                            color=cm(i / n), lw=0.9, elinewidth=0.6, capsize=1)
        tt = np.geomspace(20, 30000, 20)
        ax.plot(tt, 0.3 / tt, "k:", lw=0.9, label="∝ 1/t (power-law survival)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel("hazard: deaths / replica-time")
        ax.set_title("hazard in log time bins (≥ 2 deaths): flat = one metastable state", fontsize=8.5)
        ax.legend(fontsize=7, frameon=False)
        ax = axs[2, j]
        for i, x in enumerate(L):
            t = np.array(x["t"]); m = np.array(x["rho_mean"]); nal = np.array(x["n_alive_t"])
            ok = (m > 0) & (nal >= 5)
            ax.loglog(t[ok], m[ok], color=cm(i / n), lw=1.1, label=f"g {x['g']:.3g}")
        ax.axvline(50, color="grey", ls=":", lw=0.8); ax.axvline(200, color="grey", ls=":", lw=0.8)
        ax.set_xlabel("t"); ax.set_ylabel("mean active fraction ρ(t), all replicas (dead = 0)")
        ax.set_title("ρ(t) (pre-registered fit windows start at t = 50 and 200)", fontsize=8.5)
        ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle("Slow-window lifetimes in M2 from a fully active start (T = 30000, 128 replicas per g). Many replicas die early "
                 "(t < 200, while the resources used up by the start recover); the rest live much longer.\nPost hoc: the residual "
                 "lifetimes after t = 200 pass an exponential test at 21 of 21 points (15 of 17 after t = 500). On the null the hazard "
                 "drop is this early/late split, not a 1/t law (the late bins hold few deaths).", fontsize=9.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.04)
    fig.savefig(f"{FIG}/decay_lifetimes.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------------------------------------------ where
def fig_where(B, W):
    mo = m1_order(); m1 = m1_points()
    fig, axs = plt.subplots(1, 3, figsize=(15, 6.0))
    # M1 AL share and PR/N at the first sustained point (tau >= 3000) per r
    r1, al1, pr1 = [], [], []
    for (gk, r), v in sorted(m1.items()):
        if gk != "real":
            continue
        pts = [p for p in v["points"] if p["phase"] in "LD"]
        if not pts:
            continue
        p0 = pts[0]
        o = [x for x in mo.get(("real", r), []) if abs(x[0] / p0["g"] - 1) < 1e-4]
        r1.append(r); pr1.append(p0["PRN"]); al1.append(o[0][2] if o else np.nan)
    axs[0].plot(r1, al1, "s--", color="grey", label="M1, first sustained g")
    axs[1].plot(r1, pr1, "s--", color="grey", label="M1, first sustained g")
    for model, mk, col in (("M2", "o", "#e4572e"), ("M2s", "^", "#2a6fdb")):
        for b, ls in (("on", "-"), ("D", ":"), ("S", "-.")):
            rr, al, pr = [], [], []
            for k, w in W.items():
                e = B["rows"].get(k)
                if not e or e["model"] != model or e["graph"] != "real" or b not in w:
                    continue
                rr.append(e["r"]); al.append(num(w[b]["region_share"]["AL_module"])); pr.append(num(w[b]["PRN"]))
            if not rr:
                continue
            o = np.argsort(rr)
            lab = {"on": "first sustained g", "D": "first delocalized g", "S": "S line (occupancy t 50-400)"}[b]
            axs[0].plot(np.array(rr)[o], np.array(al)[o], mk + ls, color=col, ms=4, label=f"{model}, {lab}")
            axs[1].plot(np.array(rr)[o], np.array(pr)[o], mk + ls, color=col, ms=4, label=f"{model}, {lab}")
    axs[0].axhline(526 / N, color="black", lw=0.7, ls=":"); axs[0].text(5.9, 526 / N * 1.3, "size fraction 0.003", fontsize=7, ha="right")
    axs[0].axhline(0.25, color="black", lw=0.7, ls="--"); axs[0].text(5.9, 0.26, "'AL-led' (25%)", fontsize=7, ha="right")
    axs[0].set_yscale("log"); axs[0].set_xlabel("r"); axs[0].set_ylabel("AL-module share of activity"); axs[0].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    axs[0].set_title("Is activity AL-led at the boundary?", fontsize=9)
    axs[1].axhline(0.25, color="black", lw=0.7, ls="--"); axs[1].set_xlabel("r"); axs[1].set_ylabel("PR / N")
    axs[1].set_yscale("log"); axs[1].set_title("How many neurons carry the activity", fontsize=9)
    axs[1].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    # superclass composition at the first sustained point, M2 real R_CORE
    rows = sorted([(e["r"], k) for k, e in B["rows"].items() if e["model"] == "M2" and e["graph"] == "real"
                   and e["r"] in (0, 0.25, 0.5, 1, 2, 4, 6) and "on" in W.get(k, {})])
    names = set()
    for r, k in rows:
        names |= {s["superclass"] for s in W[k]["on"]["top_superclasses"]}
    names = sorted(names)
    cmap = plt.get_cmap("tab20")
    bottom = np.zeros(len(rows))
    for i, nm in enumerate(names):
        v = np.array([next((s["share"] for s in W[k]["on"]["top_superclasses"] if s["superclass"] == nm), 0) for r, k in rows])
        axs[2].bar(range(len(rows)), v, bottom=bottom, color=cmap(i % 20), label=nm); bottom += v
    axs[2].bar(range(len(rows)), 1 - bottom, bottom=bottom, color="#eeeeee", label="other")
    axs[2].set_xticks(range(len(rows))); axs[2].set_xticklabels([f"r={r:g}\ng={W[k]['on']['g']:.2g}" for r, k in rows], fontsize=7)
    axs[2].set_ylabel("share of activity (occupancy t > 1500)"); axs[2].set_title("M2: top-5 superclasses at the first sustained g", fontsize=9)
    axs[2].legend(fontsize=6, frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/decay_where.png", dpi=140); plt.close(fig)


# ------------------------------------------------------------------------------------------------ variants
def fig_variants(B):
    order = ["M1", "M1s", "M2", "M2s", "M2_allsyn", "M2_KW09", "M2_N15", "M2_tau30"]
    lab = {"M1": "M1 (no depression)", "M1s": "M1s (lLN1_bc per-target signs only)", "M2": "M2 (default)",
           "M2s": "M2s (M2 + per-target signs)", "M2_allsyn": "M2_allsyn (all outputs depress)",
           "M2_KW09": "M2_KW09 (U 0.28, TAU 240)", "M2_N15": "M2_N15 (U 0.22, TAU 89)", "M2_tau30": "M2_tau30 (TAU 30)"}
    rc = {0: "#1b9e77", 1: "#d95f02", 4: "#7570b3"}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for j, b in enumerate(("S", "on", "D")):
        ax = axs[j]
        for i, m in enumerate(order):
            for r, c in rc.items():
                e = B["rows"].get(f"{m}_real_r{r:g}")
                if m == "M1" and e is None:
                    continue
                if e is None:
                    continue
                g = num(e[b]["g"]); tot = e[b]["total"]
                y = i + {0: -0.22, 1: 0, 4: 0.22}[r]
                if np.isfinite(g):
                    ax.errorbar(g, y, xerr=[[g - num(tot[0])], [num(tot[1]) - g]] if tot else None, fmt="o", color=c, ms=5,
                                mec="black", mew=0.4, capsize=2, label=f"r = {r}" if (i == 2) else None)
                    if b == "D" and e[b]["status"] == "at_onset":
                        ax.plot(g, y, "D", mfc="none", mec=c, ms=9)
                elif e[b]["status"] in ("none", "no_sustained_points"):
                    ax.plot(30, y, ">", color=c, ms=6, mfc="none")
        ax.set_xscale("log"); ax.set_xlim(0.1, 40)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([lab[m] for m in order], fontsize=8)
        ax.set_title({"S": "S line (τ = 200)", "on": "sustained onset (τ = 3000)", "D": "delocalized (PR = N/4); ▷ = not by g 30"}[b], fontsize=9)
        ax.set_xlabel("coupling g"); ax.axvline(G_LIN, color="black", ls=":", lw=0.8)
        ax.invert_yaxis()
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Synapse-model variants at r = 0, 1, 4 (h = 0 decay; error bars = grid bracket ∪ bootstrap)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{FIG}/decay_variants.png", dpi=140); plt.close(fig)


# ------------------------------------------------------------------------------------------------ oscillation
def fig_osc(OS):
    rows = OS["rows"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    cols = {"total": "#444444", "AL_module": "#e4572e", "nonORN": "#2a6fdb"}
    for sig, c in cols.items():
        xs = [(num(o["c_min"]), num(o["c_peak"]), o["row"]) for o in rows if o["signal"] == sig]
        if not xs:
            continue
        a = np.array([[x[0], x[1]] for x in xs], float)
        m2 = np.array([x[2].startswith("M2") for x in xs])
        axs[0].scatter(a[m2, 0], np.nan_to_num(a[m2, 1], nan=-0.05), s=8, color=c, alpha=0.6, label=f"{sig} (M2 family)")
        axs[0].scatter(a[~m2, 0], np.nan_to_num(a[~m2, 1], nan=-0.05), s=14, marker="x", color=c, alpha=0.8, label=f"{sig} (M1, M1s)")
    axs[0].axvline(-0.1, color="black", ls="--", lw=0.8); axs[0].axhline(0.1, color="black", ls="--", lw=0.8)
    axs[0].fill_between([-1, -0.1], 0.1, 1, color="#f5d0c5", alpha=0.5, lw=0, zorder=0)
    axs[0].text(-0.12, 0.5, "OSC region\n(also needs period ≤ 375)", ha="right", fontsize=8)
    axs[0].set_xlim(-0.6, 0.05); axs[0].set_ylim(-0.08, 0.7)
    axs[0].set_xlabel("ACF trough C_min (first deep local minimum)"); axs[0].set_ylabel("ACF peak after the trough (nan → −0.05)")
    axs[0].legend(fontsize=6.5, frameon=False, loc="upper right"); axs[0].set_title("Every sustained point (≥ 4 replicas alive at T)", fontsize=9)
    # trough depth vs g for M2 real R_CORE, total signal
    cm = plt.get_cmap("viridis")
    rr = sorted({o["row"] for o in rows if o["row"].startswith("M2_real_r")}, key=lambda k: float(k.split("_r")[-1]))
    for i, k in enumerate(rr):
        xs = sorted([(o["g"], num(o["c_min"])) for o in rows if o["row"] == k and o["signal"] == "total"])
        if xs:
            axs[1].plot([x[0] for x in xs], [x[1] for x in xs], "-", color=cm(float(k.split("_r")[-1]) / 6), lw=0.9)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 6))
    fig.colorbar(sm, ax=axs[1], fraction=0.05, pad=0.02).set_label("r")
    axs[1].axhline(-0.1, color="black", ls="--", lw=0.8); axs[1].set_xscale("log")
    axs[1].set_xlabel("coupling g"); axs[1].set_ylabel("ACF trough C_min (total activity)")
    axs[1].set_title("M2 real graph, all 23 r: depth of the ACF undershoot", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/decay_osc.png", dpi=140); plt.close(fig)


# ------------------------------------------------------------------------------------------------ M2s vs M2
def fig_m2s(B, W):
    m1 = m1_points()
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.0))
    ax = axs[0]
    sty = {"M2": ("o", "-", 1.0), "M2s": ("^", "--", 1.0), "M1s": ("s", ":", 0.8)}
    for model, (mk, ls, a) in sty.items():
        for b in ("S", "on", "D"):
            rows = sorted([(e["r"], e) for e in B["rows"].values() if e["model"] == model and e["graph"] == "real"
                           and e["r"] in (0, 0.25, 0.5, 1, 2, 4, 6)], key=lambda x: x[0])
            if not rows:
                continue
            r = np.array([x[0] for x in rows]); g = np.array([num(x[1][b]["g"]) for x in rows])
            lo = np.array([num((x[1][b]["total"] or [None, None])[0]) for x in rows]); hi = np.array([num((x[1][b]["total"] or [None, None])[1]) for x in rows])
            ok = np.isfinite(g)
            ax.errorbar(r[ok] + {"M2": -0.04, "M2s": 0.04, "M1s": 0.0}[model], g[ok],
                        yerr=[np.nan_to_num(g[ok] - lo[ok]), np.nan_to_num(hi[ok] - g[ok])], fmt=mk + ls, color=B_COL[b],
                        ms=5, mfc="white" if model != "M2" else B_COL[b], alpha=a, capsize=2, lw=1,
                        label=f"{model}: {B_LAB[b]}")
    r1 = sorted(r for (gk, r) in m1 if gk == "real")
    for b in ("S", "on", "D"):
        ax.plot(r1, [num(m1[("real", r)][b]["g"]) for r in r1], color=B_COL[b], lw=4, alpha=0.18)
    ax.plot([], [], color="grey", lw=4, alpha=0.3, label="M1 (analysis2), all three lines")
    ax.set_yscale("log"); ax.set_xlabel("inhibition ratio r"); ax.set_ylabel("coupling g at the boundary")
    ax.axhline(G_LIN, color="black", ls=":", lw=0.8)
    ax.set_title("M2 vs M2s (lLN1_bc→PN 0, →LN +1) vs M1s (signs fix without depression)", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    ax = axs[1]
    for model, (mk, ls, a) in sty.items():
        rows = sorted([(e["r"], k) for k, e in B["rows"].items() if e["model"] == model and e["graph"] == "real"
                       and "on" in W.get(k, {})])
        if rows:
            ax.plot([r for r, k in rows], [num(W[k]["on"]["region_share"]["AL_module"]) for r, k in rows], mk + ls,
                    color="black", mfc="white" if model != "M2" else "black", label=model)
    mo = m1_order()
    rr, al = [], []
    for r in r1:
        pts = [p for p in m1[("real", r)]["points"] if p["phase"] in "LD"]
        o = [x for x in mo.get(("real", r), []) if pts and abs(x[0] / pts[0]["g"] - 1) < 1e-4]
        if o:
            rr.append(r); al.append(o[0][2])
    ax.plot(rr, al, "s-", color="grey", alpha=0.6, label="M1")
    ax.axhline(0.25, color="black", ls="--", lw=0.7); ax.set_yscale("log")
    ax.text(3.0, 0.265, "'AL-led' threshold (25%, plan 4.3)", fontsize=7.5, ha="center")
    ax.set_xlabel("inhibition ratio r"); ax.set_ylabel("AL-module share at the first sustained g")
    ax.set_title("Who carries the first sustained activity", fontsize=9); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/decay_m2s.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    B = jl("boundaries.json"); P = jl("points.json")
    which = sys.argv[1:] or ["tiles", "lines", "nulls", "order", "lifetimes", "where", "variants", "osc", "m2s"]
    if "tiles" in which:
        fig_tiles(P, B)
    if "lines" in which:
        fig_lines(B)
    if "nulls" in which:
        fig_nulls(P, B)
    if "order" in which:
        fig_order(P, B)
    if "lifetimes" in which and os.path.exists(f"{OUT}/lifetimes.json"):
        fig_lifetimes(jl("lifetimes.json"))
    if "where" in which:
        fig_where(B, jl("where.json"))
    if "variants" in which:
        fig_variants(B)
    if "osc" in which:
        fig_osc(jl("osc.json"))
    if "m2s" in which:
        fig_m2s(B, jl("where.json"))
    print("figures ->", FIG)
