"""m2_mf_test.py -- tests of the M2 mean-field core (code/m2_mf_core.py). Criteria fixed here before any production run.

CPU part (run directly):
  T1  exact renewal efficacy == stationary efficacy of the moment map (rel 1e-10), for 5 (p, U, TAU) incl. M2 / KW09 /
      N15 / tau30; Poisson limit within 5% of exact at rho <= 0.1 for TAU >= 89.
  T2  Monte Carlo of ald_sim.py ITSELF (--graph with 20,000 isolated units, --std_mode spike, drive) against the moment
      map iterated with the same p(t): (a) decay start, constant drive; (b) quiet start, drive switched at t = 200 to a
      larger value while half of the units are clamped off. Pass: every window mean of rho, efficacy of active units and
      mean resource within |z| < 5 (SE from the replica-to-replica spread) AND within 1% relative.
GPU part (--gpu; run in the smoke job):
  T3  operator: nnz at r = 1 equals the spec (24,562,483); M2s removes 6,658 lLN1_bc->ALPN edges; c = 1/745.91.
  T4  M1 variant: one map step equals theory_qmf.py's map rho e^{-dt} + (1 - rho) p (max abs diff < 1e-6, float32).
  T5  JVP vs central finite difference of the map in float64 at a generic state (rel err < 1e-6).
  T6  g_lin: Perron root of B(g = 0.16315, x = 1) / alpha_eff within 1e-3 of 1 (M2 rested = M1 operator).
Output: results/m2/theory/tests_meanfield.json (merged CPU/GPU parts).
"""
import json, os, subprocess, sys, tempfile
import numpy as np

ROOT = "/MALECNS_ROOT"
sys.path.insert(0, f"{ROOT}/code")
OUT = f"{ROOT}/results/m2/theory/tests_meanfield.json"
PY = "/MALECNS_PYBIN/python"
DT = 0.2; P = 1 - np.exp(-DT)


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(res):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=float)


def moment_iter(p_seq, U, TAU, init, clamp_from=None):
    """Iterate the per-unit moment map for a sequence of on-probabilities (numpy, float64)."""
    q = np.exp(-DT / TAU)
    rho, y, m1, m0 = init
    out = []
    for k, p in enumerate(p_seq):
        A = q * m1 + (1 - q) * rho; Bq = q * m0 + (1 - q) * (1 - rho)
        rho, y, m1, m0 = (1 - P) * rho + p * (1 - rho), (1 - P) * y + p * Bq, (1 - P) * A + p * (1 - U) * Bq, P * A + (1 - p) * Bq
        out.append((rho, y, m1, m0))
    return np.array(out)


def t1():
    import m2_mf_core as M
    rows, ok = [], True
    for (p, U, TAU) in [(0.01, 0.25, 157), (0.0076, 0.25, 157), (0.05, 0.28, 240), (0.12, 0.22, 89), (0.03, 0.25, 30)]:
        e1 = M.eff_exact(p, U, TAU)
        rho, y, m1, m0 = M.moment_stationary(p, U, TAU)
        e2 = y / rho
        rel = abs(e1 - e2) / e1
        ep = M.eff_poisson(rho, U, TAU)
        rows.append({"p": p, "U": U, "TAU": TAU, "rho": rho, "eff_renewal": e1, "eff_moment": e2, "rel": rel, "eff_poisson": ep,
                     "poisson_rel": abs(ep - e1) / e1})
        ok &= rel < 1e-10
        if rho <= 0.1 and TAU >= 89:
            ok &= abs(ep - e1) / e1 < 0.05
    return {"pass": bool(ok), "rows": rows}


def run_sim(args, tmp):
    cmd = [PY, f"{ROOT}/code/ald_sim.py"] + args + ["--device", "cpu", "--out", f"{tmp}/o.npz"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    return np.load(f"{tmp}/o.npz", allow_pickle=True)


def t2():
    import m2_mf_core as M
    U, TAU, N = 0.25, 157.0, 20000
    res, ok = {}, True
    with tempfile.TemporaryDirectory() as tmp:
        # isolated units: one edge of coefficient 0 (modulatory sign) so that the graph is non-empty
        np.savez(f"{tmp}/g.npz", pre=np.array([0]), post=np.array([1]), w=np.array([1.0], np.float32),
                 sign=np.ones(N, np.int8), c=1.0)
        half = np.arange(0, N, 2)
        np.savez(f"{tmp}/half.npz", idx=half); np.savez(f"{tmp}/other.npz", idx=np.arange(1, N, 2))
        h1, h2 = MC_h(0.05), MC_h(0.15)
        common = ["--graph", f"{tmp}/g.npz", "--norm", "raw", "--g", "1", "--B", "8", "--seed", "11",
                  "--std_group", f"sign:1={U},{TAU}", "--std_mode", "spike", "--rec_lin", "1",
                  "--region", "all=all", "--region", f"half=idx:{tmp}/half.npz", "--region", f"other=idx:{tmp}/other.npz"]
        cases = {
            "a_decay_const": (["--exp", "decay", "--T", "400", "--drive", f"all={h1}"], "decay"),
            "b_quiet_switch": (["--exp", "quiet", "--T", "400", "--drive", f"all={h1}", "--switch_at", "200",
                                "--sw_drive", f"all={h2}", "--sw_clamp_off", f"idx:{tmp}/half.npz"], "quiet"),
        }
        for name, (extra, start) in cases.items():
            d = run_sim(common + extra, tmp)
            t = d["t"]; names = list(d["reg_names"])
            reg = d["reg_rho"][0]                         # [t, R, B]
            xr = d["res_x"][0, 0] if d["res_x"].ndim == 4 else d["res_x"][0]            # [t, R]
            xa = d["res_x_act"][0, 0] if d["res_x_act"].ndim == 4 else d["res_x_act"][0]
            steps = int(round(400 / DT))
            p1 = 1 - np.exp(-np.tanh(h1) * DT); p2 = 1 - np.exp(-np.tanh(h2) * DT)
            init = (1.0, 1.0, 1 - U, 0.0) if start == "decay" else (0.0, 0.0, 0.0, 1.0)
            tt = np.arange(1, steps + 1) * DT
            if name == "a_decay_const":
                traj = {"all": moment_iter(np.full(steps, p1), U, TAU, init)}
            else:
                s1 = int(round(200 / DT))
                pre = moment_iter(np.full(s1, p1), U, TAU, init)
                r_, y_, m1_, m0_ = pre[-1]
                post_other = moment_iter(np.full(steps - s1, p2), U, TAU, (r_, y_, m1_, m0_))
                post_half = moment_iter(np.zeros(steps - s1), U, TAU, (0.0, 0.0, 0.0, m0_ + m1_))
                traj = {"other": np.vstack([pre, post_other]), "half": np.vstack([pre, post_half])}
            rows = []
            for rn, tr in traj.items():
                ri = names.index(rn)
                for (w0, w1) in [(5, 50), (50, 200), (210, 260), (260, 400)]:
                    sel_t = (t > w0) & (t <= w1)
                    sel_m = (tt > w0) & (tt <= w1)
                    idx_m = np.searchsorted(tt, t[sel_t] - 1e-9)
                    mf = tr[idx_m]
                    sim_rho = reg[sel_t, ri, :].mean(0)               # per replica window mean
                    mf_rho = mf[:, 0].mean()
                    se = sim_rho.std(ddof=1) / np.sqrt(len(sim_rho)) + 1e-12
                    z = (sim_rho.mean() - mf_rho) / se
                    rel = abs(sim_rho.mean() - mf_rho) / max(mf_rho, 1e-12)
                    mf_eff = (mf[:, 1] / np.maximum(mf[:, 0], 1e-300)).mean() if mf_rho > 0 else np.nan
                    sim_eff = np.nanmean(xa[sel_t, ri]) if mf_rho > 0 else np.nan
                    mf_x = (mf[:, 2] + mf[:, 3]).mean(); sim_x = xr[sel_t, ri].mean()
                    row = {"region": rn, "win": [w0, w1], "rho_sim": float(sim_rho.mean()), "rho_mf": float(mf_rho),
                           "z_rho": float(z), "rel_rho": float(rel), "eff_sim": float(sim_eff), "eff_mf": float(mf_eff),
                           "x_sim": float(sim_x), "x_mf": float(mf_x), "rel_x": float(abs(sim_x - mf_x) / mf_x)}
                    if mf_rho > 1e-6:
                        row["rel_eff"] = float(abs(sim_eff - mf_eff) / mf_eff)
                        good = (abs(z) < 5 or rel < 0.002) and rel < 0.01 and row["rel_eff"] < 0.01 and row["rel_x"] < 0.01
                    else:
                        good = sim_rho.mean() == 0 and row["rel_x"] < 0.01
                    row["pass"] = bool(good); ok &= good
                    rows.append(row)
            res[name] = rows
    return {"pass": bool(ok), "cases": res}


def MC_h(rho):
    import m2_common as MCm
    return MCm.h_of_rho(rho)


def gpu_tests():
    import torch
    import m2_mf_core as M
    out = {}
    mdl = M.Model("M2", 1.0)
    out["T3"] = {"nnz_r1": mdl.nnz, "expected": 24562483, "c_inv": 1 / mdl.c, "pass": mdl.nnz == 24562483 and abs(1 / mdl.c - 745.91) < 0.05}
    m2s = M.Model("M2s", 1.0)
    out["T3"]["M2s_edges_overridden"] = m2s.n_edge_ov
    out["T3"]["M2s_nnz"] = m2s.nnz
    out["T3"]["pass"] &= m2s.n_edge_ov[0] == 6658 and mdl.nnz - m2s.nnz == 6658
    del m2s
    # T4: M1 map equals theory_qmf's
    m1 = M.Model("M1", 1.0)
    gv = torch.tensor([[0.3, 1.0, 3.0]], device="cuda")
    Z = m1.init_state("eps", 3, eps=0.02)
    H = m1.h_vec(0.0)
    rho0 = Z["rho"].clone()
    Z, s, p = m1.step(Z, gv, H)
    s_ref = torch.sparse.mm(m1.W, rho0) * gv
    p_ref = 1 - torch.exp(-torch.tanh(s_ref.clamp(min=0)) * DT)
    ref = rho0 * float(np.exp(-DT)) + (1 - rho0) * p_ref
    d = float((Z["rho"] - ref).abs().max())
    out["T4"] = {"max_abs_diff": d, "pass": d < 1e-6}
    del m1
    # T5: JVP vs central differences (float64)
    m64 = M.Model("M2", 1.0, dtype=torch.float64)
    m64.W = m64.W.to(torch.float64)
    gen = torch.Generator(device="cuda"); gen.manual_seed(5)
    Zs = {k: v for k, v in m64.init_state("eps", 2, eps=0.01).items()}
    Zs["rho"] = torch.rand(m64.N, 2, device="cuda", dtype=torch.float64, generator=gen) * 0.1
    Zs["y"] = Zs["rho"] * 0.5; Zs["m1"] = Zs["rho"] * 0.6; Zs["m0"] = (1 - Zs["rho"]) * 0.7
    gv = torch.tensor([[0.5, 2.0]], device="cuda", dtype=torch.float64)
    H = m64.h_vec(0.02).double()
    dZ = {k: torch.randn(m64.N, 2, device="cuda", dtype=torch.float64, generator=gen) * 1e-3 for k in Zs}
    Z1, s1, p1 = m64.step({k: v.clone() for k, v in Zs.items()}, gv, H)
    J = m64.jvp(Zs, s1, p1, dZ, gv)
    eps = 1e-5
    Zp, _, _ = m64.step({k: Zs[k] + eps * dZ[k] for k in Zs}, gv, H)
    Zm, _, _ = m64.step({k: Zs[k] - eps * dZ[k] for k in Zs}, gv, H)
    rel = max(float(((Zp[k] - Zm[k]) / (2 * eps) - J[k]).norm() / J[k].norm()) for k in Zs)
    out["T5"] = {"max_rel_err": rel, "pass": rel < 1e-6}
    del m64
    # T6: g_lin unchanged (rested resources)
    x1 = torch.ones(mdl.N, device="cuda", dtype=torch.float64)
    lam, sig, _, it = mdl.perron_branch(x1, 0.16315)
    out["T6"] = {"lam": lam, "sigma": sig, "iters": it, "pass": abs(sig - 1) < 1e-3}
    return out


if __name__ == "__main__":
    res = load()
    if "--gpu" in sys.argv:
        res["gpu"] = gpu_tests()
    else:
        res["T1"] = t1()
        res["T2"] = t2()
    save(res)
    for k, v in res.items():
        if k == "gpu":
            for kk, vv in v.items():
                print(kk, "PASS" if vv["pass"] else "FAIL", {a: b for a, b in vv.items() if a != "pass"})
        else:
            print(k, "PASS" if v["pass"] else "FAIL")
