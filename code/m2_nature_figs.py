"""m2_nature_figs.py -- figures of lane m2/nature (figs/m2/nature_*.png), from the JSON files in results/m2/nature/.

  python m2_nature_figs.py [fss] [hyst] [aval] [osc]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/m2/nature"
FIG = f"{ROOT}/figs/m2"
# reference categorical palette (dataviz skill, light mode), fixed order
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
     "green": "#008300", "violet": "#6250d6", "red": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAM = {"cnull": ("configuration null", C["blue"], "o"), "realsubu": ("real, uniform subsample (c renorm.)", C["orange"], "s"),
       "realsubcn": ("real, AL kept (c renorm.)", C["aqua"], "^"), "realsub": ("real, AL kept (c fixed)", C["violet"], "D")}
MOD = {"M1": ("M1 (no depression)", C["orange"]), "M2": ("M2 (depression + ORN drive)", C["blue"])}
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.facecolor": "white", "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left"})


def seq_blues(n):
    return [plt.cm.Blues(x) for x in np.linspace(0.35, 0.95, n)]


# ------------------------------------------------------------------------------------------------ FSS
def fig_fss():
    F = json.load(open(f"{OUT}/fss.json"))
    fams = F["families"]; per = F["per_graph"]
    fig, axs = plt.subplots(3, 3, figsize=(13.5, 11.5))
    for j, r in enumerate((0.0, 1.0, 4.0)):
        axW, axO = axs[0, j], axs[1, j]
        vtxt = []
        for fam, (lab, col, mk) in FAM.items():
            s = fams.get(f"{fam}|r{r:g}")
            if not s:
                continue
            rows = [x for x in s["sizes"] if x["W_mean"] is not None]
            N = [x["N"] for x in rows]
            for x in rows:                                  # individual seeds
                axW.plot([x["N"]] * len(x["W"]), x["W"], mk, color=col, ms=4, alpha=0.35, mec="none")
                axO.plot([x["N"]] * len(x["on"]), x["on"], mk, color=col, ms=4, alpha=0.35, mec="none")
            axW.plot(N, [x["W_mean"] for x in rows], "-" + mk, color=col, lw=2, ms=7, mec="white", mew=1.2,
                     label=(lab if j == 0 else None))
            vtxt.append((col, f"{lab.split(',')[0] if fam == 'cnull' else lab}: {s['decision']['verdict']}"))
            axO.plot(N, [x["on_mean"] for x in rows], "-" + mk, color=col, lw=2, ms=7, mec="white", mew=1.2, label=lab)
        for ax in (axW, axO):
            ax.set_xscale("log"); ax.set_xlabel("network size N (neurons)")
        axW.set_ylim(bottom=0)
        axW.set_ylabel("crossover width W = ln(g_onset / g_S)")
        axW.set_title(f"{'ABC'[j]}  r = {r:g}: width of the S -> onset crossover")
        axO.set_ylabel("onset g (lifetime = 3000)"); axO.set_title(f"{'DEF'[j]}  r = {r:g}: where the onset sits")
        for k, (col, t) in enumerate(vtxt):
            axW.text(0.02, 0.03 + 0.055 * k, t, transform=axW.transAxes, fontsize=7, color=col, va="bottom")
        axW.set_ylim(0, axW.get_ylim()[1] * 1.05)
    # lifetime curves at r = 1: cnull s1 and realsubu s1 by N (+ full)
    for k, (fam, ttl) in enumerate((("cnull", "G  configuration nulls, r = 1"), ("realsubu", "H  real, uniform subsample, r = 1"))):
        ax = axs[2, k]
        recs = sorted([x for x in per if x["r"] == 1.0 and x["gseed"] in (0, 1) and
                       (x["family"] == fam or (fam != "cnull" and x["family"] == "real"))], key=lambda x: (x["N"], -x["gseed"]))
        recs = [x for i, x in enumerate(recs) if i == 0 or x["N"] != recs[i - 1]["N"]]      # one seed per size
        cols = seq_blues(len(recs)) if fam == "cnull" else [plt.cm.Oranges(v) for v in np.linspace(0.4, 0.95, len(recs))]
        for x, cl in zip(recs, cols):
            P = [p for p in x["points"]]
            g = [p["g"] for p in P]; tau = [p["tau"] if p["tau"] is not None else 3e5 for p in P]
            ax.plot(g, tau, "-o", color=cl, ms=3, lw=1.5, label=f"N = {x['N']:,}" + (" (full brain)" if x["family"] == "real" else ""))
        ax.axhline(200, color=INK2, ls=":", lw=1); ax.axhline(3000, color=INK2, ls=":", lw=1)
        ax.text(ax.get_xlim()[0] if False else 0.72, 230, "tau = 200 (S)", fontsize=7.5, color=INK2)
        ax.text(0.72, 3400, "tau = 3000 (onset)", fontsize=7.5, color=INK2)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.7, 6); ax.set_ylim(10, 4e5)
        ax.set_xlabel("g"); ax.set_ylabel("lifetime tau (model time; 3e5 = no death)"); ax.set_title(ttl)
        ax.legend(fontsize=7.5, loc="lower right")
    # chi_max of survivors vs N
    ax = axs[2, 2]
    for fam, (lab, col, mk) in FAM.items():
        for r, ls in ((0.0, "-"), (1.0, "--")):
            s = fams.get(f"{fam}|r{r:g}")
            if not s:
                continue
            rows = [x for x in s["sizes"] if x.get("chi_max_mean")]
            ax.plot([x["N"] for x in rows], [x["chi_max_mean"] for x in rows], ls + mk, color=col, ms=6, mec="white",
                    label=f"{lab}, r = {r:g}" if r == 0 else None)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("N"); ax.set_ylabel("max over g of N var_t(rho) (survivors)")
    ax.set_title("I  max of N var_t(rho), survivors, g <= 11.2\n(solid r 0, dashed r 1)")
    ax.legend(fontsize=7)
    fig.suptitle("M2, setting (a): finite-size scaling of the onset crossover (h = 0, fully active start, T = 3000, B = 64)",
                 fontsize=12, x=0.01, ha="left")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=4, fontsize=8.5, bbox_to_anchor=(0.01, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(f"{FIG}/nature_fss.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------------------------------------------ hysteresis
def fig_hyst():
    H = json.load(open(f"{OUT}/hyst.json"))
    runs = H["runs"]; ver = H["verdicts"]
    fig, axs = plt.subplots(3, 4, figsize=(15, 11))
    Dcol = {50: C["yellow"], 200: C["aqua"], 800: C["blue"], 3200: C["violet"]}
    ex = [("ramp_h0_down", "M2", 1), ("ramp_h0_down", "M2", 0), ("ramp_h0_down", "M1", 1), ("ramp_driven", "M2", 1)]
    for k, (proto, model, r) in enumerate(ex):
        ax = axs[0, k]
        for D in (50, 200, 800, 3200):
            rr = [x for x in runs if x["proto"] == proto and x["model"] == model and x["r"] == r and x["D"] == D and x["rep"] == 0]
            if not rr:
                continue
            x = rr[0]; g = np.array(x["g"]) / x["g_on"]
            first = "down (from the active start)" if proto == "ramp_h0_down" else "up"
            ax.plot(g, x["x_first"], "-", color=Dcol[D], lw=2, label=f"D = {D}: {first}")
            ax.plot(g, x["x_second"], "--", color=Dcol[D], lw=2, label=f"D = {D}: back")
        ax.axvline(1, color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("g / onset g (decay lane)")
        ax.set_ylabel("nonORN active fraction" if proto == "ramp_driven" else "active fraction")
        ax.set_title(f"{'ABCD'[k]}  {model} {'h = 0 ramp' if proto == 'ramp_h0_down' else 'ORN-driven ramp'}, r = {r}")
        if k == 0:
            ax.legend(fontsize=6.5, loc="upper left")
    # A(D) per r
    for i, (proto, model) in enumerate([("ramp_h0_down", "M2"), ("ramp_h0_down", "M1"), ("ramp_driven", "M2"),
                                        ("ramp_driven", "M1")]):
        ax = axs[1, i]
        cols = [plt.cm.viridis(v) for v in np.linspace(0, 0.9, 7)]
        for j, r in enumerate((0, 0.25, 0.5, 1, 2, 4, 6)):
            v = ver.get(f"{proto}|{model}|r{r:g}")
            if not v:
                continue
            Ds = sorted(int(d) for d in v["A"]); A = [v["A"][str(d)] if str(d) in v["A"] else v["A"][d] for d in Ds]
            ax.plot(Ds, A, "-o", color=cols[j], ms=5, lw=1.6, label=f"r = {r:g}")
            ph = v.get("posthoc_all_runs")
            if ph:                                   # post-hoc runs (extra seeds, D = 3200): open markers
                Dp_ = sorted(int(d) for d in ph["A_reps"])
                for d in Dp_:
                    for a_ in ph["A_reps"][str(d)]:
                        ax.plot([d], [a_], "o", mfc="none", mec=cols[j], ms=7, mew=1.3)
            p99 = v["p99"]; Dp = sorted(int(d) for d in p99)
            ax.plot(Dp, [p99[str(d)] if str(d) in p99 else p99[d] for d in Dp], ":", color=cols[j], lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("dwell D per g (model time)")
        ax.set_ylabel("loop area A (dotted: 99th pct of permutations)")
        ax.set_title(f"{'EFGH'[i]}  {model}, {'h = 0' if proto == 'ramp_h0_down' else 'ORN-driven'}: area vs dwell")
        if i == 0:
            ax.legend(fontsize=7, ncol=2)
    # resource lag and bimodality / verdict table
    ax = axs[2, 0]
    for D in (50, 200, 800, 3200):
        rr = [x for x in runs if x["proto"] == "ramp_h0_down" and x["model"] == "M2" and x["r"] == 1 and x["D"] == D and x["rep"] == 0]
        if rr and rr[0]["resource"]:
            x = rr[0]; sg = np.array(x["seg_g"]) / x["g_on"]; res = x["resource"].get("CNS_rest")
            n = len(sg); h = n // 2
            ax.plot(sg[:h + 1], res[:h + 1], "-", color=Dcol[D], lw=2, label=f"D = {D} down")
            ax.plot(sg[h:], res[h:], "--", color=Dcol[D], lw=2)
    ax.set_xscale("log"); ax.set_xlabel("g / onset g"); ax.set_ylabel("mean resource x, CNS_rest (dwell end)")
    ax.set_title("I  M2 h = 0, r = 1: resource lag\n(solid down, dashed back up)"); ax.legend(fontsize=7)
    ax = axs[2, 1]
    for j, (proto, model, col) in enumerate([("ramp_h0_down", "M2", C["blue"]), ("ramp_h0_down", "M1", C["orange"]),
                                             ("ramp_driven", "M2", C["aqua"]), ("ramp_driven", "M1", C["violet"])]):
        rs, rat = [], []
        for r in (0, 0.25, 0.5, 1, 2, 4, 6):
            v = ver.get(f"{proto}|{model}|r{r:g}")
            if v and v.get("ratio_800_200") is not None:
                rs.append(r); rat.append(v["ratio_800_200"])
        ax.plot(rs, rat, "-o", color=col, ms=6, lw=1.8, label=f"{model} {'h = 0' if proto == 'ramp_h0_down' else 'driven'}")
    ax.axhline(0.7, color=C["red"], ls="--", lw=1); ax.axhline(0.5, color=INK2, ls=":", lw=1)
    ax.text(0.05, 0.72, "first-order needs >= 0.7", fontsize=7.5, color=C["red"])
    ax.text(0.05, 0.43, "finite-rate if < 0.5", fontsize=7.5, color=INK2)
    ax.set_xlabel("r"); ax.set_ylabel("A(800) / A(200)"); ax.set_title("J  does the loop survive slower ramps?")
    ax.legend(fontsize=7)
    for k, proto in enumerate(("ramp_h0_down", "ramp_driven")):
        ax = axs[2, 2 + k]; ax.axis("off")
        lines = [f"{'h = 0 ramps' if k == 0 else 'ORN-driven ramps'}: pre-registered verdict", "  (post hoc, all runs)",
                 "model r     verdict       A800/A200  A3200/A800"]
        for key in sorted(ver):
            v = ver[key]; pr, model, r = key.split("|")
            if pr != proto:
                continue
            f2 = lambda x: "  -  " if x is None else f"{x:.2f}"
            ph = v.get("posthoc_all_runs")
            lines.append(f"{model}  {r[1:]:5s} {v['verdict'][:12]:12s}  {f2(v['ratio_800_200']):>8s}  {f2(v['ratio_3200_800']):>8s}")
            if ph:
                lines.append(f"          ({ph['verdict'][:12]:12s}  {f2(ph['ratio_800_200']):>8s}  {f2(ph['ratio_3200_800']):>8s})")
        ax.text(0, 1, "\n".join(lines), family="monospace", fontsize=7.4, va="top", color=INK)
    fig.suptitle("M2 vs M1: hysteresis from quasi-static g ramps (B = 32; same replicas up and down)", fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{FIG}/nature_hysteresis.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------------------------------------------ avalanches
def fig_aval():
    A = json.load(open(f"{OUT}/aval.json"))
    pts = A["points"]; cells = A["cells"]
    gon = {}
    for p in json.load(open(f"{OUT}/index_aval.json")):
        gon[(p["model"], float(p["r"]))] = p["g_on"]
    fig, axs = plt.subplots(3, 4, figsize=(16, 12.5))
    rs = (0, 0.25, 0.5, 1, 2, 4, 6)
    rcol = [plt.cm.viridis(v) for v in np.linspace(0, 0.9, len(rs))]
    # A, B: P(S) at r = 1, gap 0
    for k, model in enumerate(("M2", "M1")):
        ax = axs[0, k]
        ps = sorted([p for p in pts if p["model"] == model and p["r"] == 1 and p["G"] == 0 and p.get("hist_S")],
                    key=lambda p: p["g"])
        sel = ps[::max(1, len(ps) // 7)]
        cols = seq_blues(len(sel)) if model == "M2" else [plt.cm.Oranges(v) for v in np.linspace(0.4, 0.95, len(sel))]
        for p, cl in zip(sel, cols):
            e = np.array(p["hist_S"]["edges"]); dns = np.array(p["hist_S"]["dens"]); c = np.sqrt(e[:-1] * e[1:])
            ok = dns > 0
            ax.plot(c[ok], dns[ok], "-o", ms=2.5, lw=1.3, color=cl,
                    label=f"g = {p['g']:.3g} ({p['g'] / gon[(model, 1.0)]:.2f} x onset)")
        xx = np.array([1, 1e6]); ax.plot(xx, 0.3 * xx ** -1.5, ":", color=INK, lw=1.2, label="slope -3/2 (MF-DP)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("avalanche size S (onsets)"); ax.set_ylabel("P(S)")
        ax.set_title(f"{'AB'[k]}  {model}, r = 1, pulse gap 0: size distributions")
        ax.legend(fontsize=6.5, loc="lower left"); ax.set_ylim(1e-12, 2)
    # C: tau_S vs g / onset (gap 0); D: KS vs g / onset (gap 0)
    for k, (key, yl, ttl) in enumerate((("tau_S", "size exponent tau_S", "C  fitted size exponent (gap 0)"),
                                        ("ks", "KS distance of the size fit", "D  goodness of the power law (gap 0)"))):
        ax = axs[0, 2 + k]
        for model, ls in (("M2", "-"), ("M1", "--")):
            for j, r in enumerate(rs):
                c = cells.get(f"r{r:g}|G0", {}).get(model)
                if not c:
                    continue
                x = np.array(c["g"]) / gon[(model, float(r))]
                y = np.array([np.nan if v is None else v for v in c[key]], float)
                ok = np.array([(n or 0) >= 1000 for n in c["n_tail"]])
                ax.plot(x, y, ls, color=rcol[j], lw=1.2, alpha=0.8, label=(f"r = {r:g}" if model == "M2" else None))
                ax.plot(x[ok], y[ok], "o", color=rcol[j], ms=3.5, mec="none")
        if key == "tau_S":
            ax.axhspan(1.3, 1.7, color=C["aqua"], alpha=0.15, lw=0); ax.set_ylim(1, 4)
            ax.text(0.31, 1.33, "critical-like band 1.3-1.7", fontsize=7.5, color=INK2)
        else:
            ax.axhline(0.03, color=C["red"], ls="--", lw=1); ax.set_yscale("log"); ax.set_ylim(3e-3, 1)
            ax.text(0.31, 0.034, "critical-like needs KS <= 0.03", fontsize=7.5, color=C["red"])
        ax.set_xscale("log"); ax.set_xlabel("g / onset g (M2 solid, M1 dashed; dots n_tail >= 1000)", fontsize=8.5)
        ax.set_ylabel(yl); ax.set_title(ttl)
        if k == 0:
            ax.legend(fontsize=7, ncol=2, loc="upper right")
    # E: mean size; F: branching ratio (gap 0), vs g / onset
    for k, (key, yl, ttl, lg) in enumerate((("mean_size", "mean avalanche size", "E  mean size (gap 0)", True),
                                            ("m_unit", "MR branching ratio per model time unit", "F  branching ratio (gap 0)", False))):
        ax = axs[1, k]
        for model, ls in (("M2", "-"), ("M1", "--")):
            for j, r in enumerate(rs):
                c = cells.get(f"r{r:g}|G0", {}).get(model)
                if c:
                    ax.plot(np.array(c["g"]) / gon[(model, float(r))], [np.nan if v is None else v for v in c[key]],
                            ls + "o", ms=2.5, lw=1.2, color=rcol[j])
        ax.set_xscale("log"); ax.set_xlabel("g / onset g (M2 solid, M1 dashed)"); ax.set_ylabel(yl); ax.set_title(ttl)
        if lg:
            ax.set_yscale("log")
        else:
            ax.axhline(1, color=INK2, lw=0.8, ls=":")
    # G: gamma vs g/onset with crackling prediction
    ax = axs[1, 2]
    for model, ls, col in (("M2", "-", C["blue"]), ("M1", "--", C["orange"])):
        for j, r in enumerate((1,)):
            c = cells.get(f"r{r:g}|G0", {}).get(model)
            if c:
                x = np.array(c["g"]) / gon[(model, float(r))]
                ax.plot(x, [np.nan if v is None else v for v in c["gamma"]], ls + "o", color=col, ms=3.5, lw=1.5,
                        label=f"{model} <S>(T) slope gamma")
                ax.plot(x, [np.nan if v is None else v for v in c["gamma_pred"]], ls + "s", color=col, ms=3, lw=0.8,
                        alpha=0.6, label=f"{model} (tau_T-1)/(tau_S-1)")
    ax.axhline(2, color=C["red"], ls=":", lw=1); ax.text(0.31, 2.03, "MF-DP: 2", fontsize=7.5, color=C["red"])
    ax.set_xscale("log"); ax.set_ylim(0, 3.5); ax.set_xlabel("g / onset g"); ax.set_ylabel("exponent")
    ax.set_title("G  r = 1, gap 0: gamma and the crackling prediction"); ax.legend(fontsize=7)
    # H: text summary
    ax = axs[1, 3]; ax.axis("off")
    lines = ["Pre-registered LHG test (21 cells: 7 r x gaps 0/10/100)",
             f"  overall: {A['overall']}  {A['verdict_counts']}",
             "  critical-like points: M1 0, M2 0", "",
             "POST HOC: span of g with tau_S in [1.3, 1.7],", "xmin <= 10, n_tail >= 1000 (KS ignored)",
             "  r     gap0 M1/M2     gap10 M1/M2"]
    for r in rs:
        def sp(G, m):
            v = cells.get(f"r{r:g}|G{G}", {}).get(m, {}).get("posthoc_exp_only_span_run")
            return "-" if v is None else f"{v:.2f}"
        lines.append(f"  {r:<5g} {sp(0, 'M1'):>5s}/{sp(0, 'M2'):<5s}     {sp(10, 'M1'):>5s}/{sp(10, 'M2'):<5s}")
    lines += ["", "POST HOC: first g / onset with a large-", "avalanche bump (gap 0), M1 / M2"]
    for r in rs:
        def fb(m):
            c = cells.get(f"r{r:g}|G0", {}).get(m, {})
            v = c.get("posthoc_first_bump_g")
            return "none" if v is None else f"{v / gon[(m, float(r))]:.2f}"
        lines.append(f"  r = {r:<5g} {fb('M1'):>6s} / {fb('M2')}")
    ax.text(0, 1, "\n".join(lines), family="monospace", fontsize=7.6, va="top", color=INK)
    # I, J: driven regime
    D = json.load(open(f"{OUT}/driven_branch.json"))["rows"] if os.path.exists(f"{OUT}/driven_branch.json") else {}
    ax = axs[2, 0]
    for j, r in enumerate(rs):
        for model, ls in (("M2", "-"), ("M1", "--")):
            v = D.get(f"{model}|real|r{r:g}")
            if v:
                ax.plot(v["g"], [np.nan if x is None else x for x in v["m_unit"]], ls, color=rcol[j], lw=1.4,
                        label=(f"r = {r:g}" if model == "M2" else None))
    ax.axhspan(0.99, 1.01, color=C["aqua"], alpha=0.3, lw=0)
    ax.text(0.06, 0.965, "marker band |1 - m| < 0.01", fontsize=7.5, color=INK2)
    ax.set_xscale("log"); ax.set_ylim(0.2, 1.05); ax.set_xlabel("g"); ax.set_ylabel("MR branching ratio per unit (nonORN onsets)")
    ax.set_title("I  ORN-driven: m vs g (M2 solid, M1 dashed)"); ax.legend(fontsize=7, ncol=2)
    ax = axs[2, 1]
    for model, mk, col in (("M2", "o", C["blue"]), ("M1", "s", C["orange"])):
        xs, ys = [], []
        for r in rs:
            v = D.get(f"{model}|real|r{r:g}")
            if v:
                mu = np.array([np.nan if x is None else x for x in v["m_unit"]], float)
                xs.append(r); ys.append(np.nanmax(mu))
        ax.plot(xs, ys, "-" + mk, color=col, ms=6, lw=1.8, label=f"{MOD[model][0]}")
    for k in ("cnull", "wnull"):
        xs, ys = [], []
        for r in rs:
            v = D.get(f"M2|{k}|r{r:g}")
            if v:
                xs.append(r); ys.append(np.nanmax(np.array([np.nan if x is None else x for x in v["m_unit"]], float)))
        ax.plot(xs, ys, ":" + ("^" if k == "cnull" else "D"), color=C["aqua"] if k == "cnull" else C["violet"], ms=5,
                label=f"M2 {'configuration' if k == 'cnull' else 'weight-shuffled'} null")
    ax.axhline(1, color=INK2, ls=":", lw=1)
    ax.set_xlabel("r"); ax.set_ylabel("max over g of m per unit"); ax.set_ylim(0.5, 1.02)
    ax.set_title("J  ORN-driven: closest approach to m = 1"); ax.legend(fontsize=7)
    # K: P(T) M2 r1 gap0 ; L: P(S) gap 100 M2 r1 (few avalanches)
    for k, (model, G) in enumerate((("M2", 0), ("M2", 100))):
        ax = axs[2, 2 + k]
        key = "hist_T" if G == 0 else "hist_S"
        ps = sorted([p for p in pts if p["model"] == model and p["r"] == 1 and p["G"] == G and p.get(key)],
                    key=lambda p: p["g"])
        sel = ps[::max(1, len(ps) // 6)]
        for p, cl in zip(sel, seq_blues(len(sel))):
            e = np.array(p[key]["edges"]); dns = np.array(p[key]["dens"]); c = np.sqrt(e[:-1] * e[1:]); ok = dns > 0
            ax.plot(c[ok], dns[ok], "-o", ms=2.5, lw=1.2, color=cl, label=f"g = {p['g']:.3g} (n = {p['n_aval']})")
        xx = np.array([1, 1e4])
        if G == 0:
            ax.plot(xx, 0.5 * xx ** -2.0, ":", color=INK, lw=1.2, label="slope -2 (MF-DP)")
            ax.set_xlabel("duration T (steps of 0.2)"); ax.set_ylabel("P(T)")
            ax.set_title("K  M2, r = 1, gap 0: durations")
        else:
            ax.plot(xx, 0.3 * xx ** -1.5, ":", color=INK, lw=1.2, label="slope -3/2")
            ax.set_xlabel("size S"); ax.set_ylabel("P(S)")
            ax.set_title("L  M2, r = 1, gap 100: too few avalanches")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=6.5, loc="lower left")
    fig.suptitle("Avalanches: pulse protocol (h = 0, one ORN seeded when silent; B 64, T 3000) and the ORN-driven regime",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{FIG}/nature_avalanches.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------------------------------------------ oscillations
def fig_osc():
    O_ = json.load(open(f"{OUT}/osc.json"))
    rows = O_["rows"]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.5))
    rs = (0, 0.25, 0.5, 1, 2, 4, 6)
    cols = [plt.cm.viridis(v) for v in np.linspace(0, 0.9, len(rs))]
    for j, r in enumerate(rs):
        L = rows.get(f"M2_real_r{r:g}", {}).get("points", [])
        L = [p for p in L if p["n_rep"] >= 4]
        if not L:
            continue
        g = np.array([p["g"] for p in L]); osc = np.array([p["osc"] for p in L])
        for ax, key in ((axs[0, 0], "period"), (axs[0, 1], "coh"), (axs[0, 2], "cv_t"), (axs[1, 0], "amp"),
                        (axs[1, 1], "c_min")):
            y = np.array([np.nan if p[key] is None else p[key] for p in L], float)
            if key == "coh":
                y[np.array([(p["c_peak"] or 0) < 0.05 for p in L])] = np.nan     # ratio undefined without a peak
            ax.plot(g, y, "-", color=cols[j], lw=1, alpha=0.6, label=f"r = {r:g}")
            ax.plot(g[osc], y[osc], "o", color=cols[j], ms=5, mec="white")
    labs = {(0, 0): ("ACF period P (model time)", "A  period (dots = OSC flag)"),
            (0, 1): ("coherence C(2P)/C(P)", "B  coherence (OSC-strong needs >= 0.7)"),
            (0, 2): ("CV_t of the active fraction", "C  relative amplitude (bursting rule: >= 0.3)"),
            (1, 0): ("std_t of the active fraction", "D  absolute amplitude"),
            (1, 1): ("ACF first deep trough C_min", "E  ACF trough (OSC needs <= -0.1)")}
    for (i, j), (yl, tt) in labs.items():
        ax = axs[i, j]; ax.set_xscale("log"); ax.set_xlabel("g"); ax.set_ylabel(yl); ax.set_title(tt)
    axs[0, 1].axhline(0.7, color=C["red"], ls="--", lw=1); axs[0, 2].axhline(0.3, color=INK2, ls=":", lw=1)
    axs[0, 1].set_xlim(1, 35); axs[0, 1].set_ylim(-0.6, 1.05); axs[0, 1].set_title("B  coherence C(2P)/C(P) where C(P) >= 0.05\n(OSC-strong needs >= 0.7)")
    axs[1, 1].axhline(-0.1, color=INK2, ls=":", lw=1)
    axs[1, 0].set_yscale("log"); axs[0, 0].legend(fontsize=7, ncol=2)
    ax = axs[1, 2]
    for k, (fam, (lab, col, mk)) in enumerate(FAM.items()):
        for r, ls in ((0.0, "-"), (1.0, "--"), (4.0, ":")):
            v = O_["fss"].get(f"{fam}|r{r:g}")
            if v and v["slopes"]:
                ax.plot([x[0] for x in v["slopes"]], [x[1] for x in v["slopes"]], ls + mk, color=col, ms=5,
                        alpha=(1.0 if fam in ("cnull", "realsubu") else 0.45), label=(f"{lab}" if r == 0 else None))
    ax.axhline(-1, color=INK2, ls=":", lw=1); ax.axhline(0, color=INK2, ls=":", lw=1)
    ax.axhspan(-0.7, -0.3, color="#d9d8d3", alpha=0.4, lw=0)
    ax.set_xscale("log"); ax.set_xlabel("matched QS density rho_QS"); ax.set_ylabel("slope of ln var_t vs ln N")
    ax.set_title("F  slope of var_t vs N at matched rho_QS\n(-1 noise, 0 macro; r0 solid, r1 dash, r4 dot)")
    ax.legend(fontsize=6.5)
    fig.suptitle("M2 oscillations in the h = 0 quasi-stationary state (lane decay data, window 1500-3000; FSS from this lane)",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{FIG}/nature_osc.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["fss", "hyst"]
    for w in which:
        {"fss": fig_fss, "hyst": fig_hyst, "aval": fig_aval, "osc": fig_osc}[w]()
        print("fig", w)
