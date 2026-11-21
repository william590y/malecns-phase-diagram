"""ald_sim.py -- extended stochastic Wilson-Cowan simulator for the "is the AL a model defect?" study (2026-11-09).

Same model, units and conventions as code/swc.py (which is NOT modified; this file copies what it needs):
  one binary unit per neuron; active -> quiescent at rate alpha = 1 (the time unit); quiescent -> active at rate
  beta * tanh(s_i)_+ ; synchronous steps of dt (default 0.2) with p_off = 1 - exp(-dt), p_on = 1 - exp(-beta tanh(s)_+ dt);
      s_i = g * sum_j c_ij * w_ij * sigma_j * a_j * x_j  +  h_i  -  theta_i
  sigma_j = +1 for ACh, -r for GABA/Glu/His, 0 for DA/OA/5-HT/unclear/unknown (consensus NT of the release);
  c = 1 / (mean in-strength over neurons with input)  ("--norm raw", the main setting) or 1/in-strength_i ("--norm in").
  x_j = 1 and theta_i = 0 unless short-term depression / adaptation are switched on; h_i = h unless --drive is given.
With every new option off the program performs exactly the operations of swc.py in the same order with the same RNG
calls, so the same --seed gives bit-identical rho/surv/occ (checked by code/ald_validate.py).

NEW OPTIONS
 Populations are named with SELECTORS (used by --drive, --clamp_off, --clamp_on, --sign_override, --std_group,
 --adapt_group, --region):
     selector := clause (';' clause)*          union
     clause   := term ('&' term)*              intersection
     term     := ['~'] atom                    '~' = complement
     atom     := all | none
               | module      the 526-neuron AL module (results/module/module_graph.npz idx; 99% supports of modes 1-2)
               | al          AL classes: olfactory, hygrosensory, thermosensory, ALPN, ALLN, ALIN, ALON (al_common.py)
               | al_rest     al & ~module          | cns_rest   ~(al | module)
               | type:<re> | class:<re> | superclass:<re> | subclass:<re> | instance:<re> | nt:<re> | side:<re>
                             (regex FULL match on the annotation string; nt = consensus NT, e.g. acetylcholine)
               | sign:<+1|-1|0>   consensus model sign
               | idx:<file.npz>[:key]  or any token ending in .npz: neuron indices (key 'idx', default) or a bool mask
               | <bare token>  neurons whose type, class or superclass EQUALS the token or fully matches it as a regex
     examples: "olfactory" (all 2,639 ORNs = class olfactory), "lLN1_bc", "type:lLN2.*", "al&sign:+1",
               "class:olfactory&~type:ORN_(VA4|VC1|VC2|VA7l|VM7d|VM7v)"  (antennal ORNs only; the 6 palp glomeruli kept out)
     An empty selection is an error (catches typos).

 1. Drive / clamps
   --h H                  background drive for every neuron (as in swc.py).
   --drive "SEL=H"        (repeatable) set h_i = H on SEL (later flags win). An isolated driven unit has stationary
                          activity rho = p_on/(p_on + p_off) with p_on = 1 - exp(-beta tanh(H) dt).
   --clamp_off "SEL"      (repeatable) hold SEL at 0 at all times ("antenna removal"); counted in rho as inactive.
   --clamp_on "SEL"       (repeatable) hold SEL at 1 at all times (tonic source; applied after clamp_off).
 2. Signs
   --sign_override "SEL=V" (repeatable, later wins) set sigma_j for presynaptic SEL: V >= 0 is used as is (+1 exc,
                          0 silent); V < 0 means inhibitory and is scaled by r (sigma = V*r; V = -1 -> -r).
   --sign_ensemble K      K independent draws of every neuron's sign from the release's NT predictions (raw file
                          body-neurotransmitters-male-cns-v1.0.feather, joined on bodyId; cached in results/aldef/nt_table.npz).
                          The release gives ONE predicted NT and its confidence per body (and per cell type), plus
                          consensus_nt and ground_truth -- no full probability vector. Mapping used:
                            L_i = consensus_nt (the label the deterministic model uses); q_i = confidence of the tier
                            prediction that agrees with L_i (--ens_conf body: body-level predicted_nt_confidence first,
                            then the cell-type confidence; celltype: the other order); q_i = --ens_qmiss (0.5) when no
                            prediction agrees with L_i or the confidence is missing.
                            p_i(nt) = q_i [nt = L_i] + (1 - q_i) pi(nt) / (1 - pi(L_i))  for nt != L_i, where pi = release-wide
                            consensus-NT frequencies over the 8 classes ACh, GABA, Glu, His, DA, OA, 5-HT, unclear.
                            P(sigma=+1) = p(ACh); P(sigma=-1) = p(GABA)+p(Glu)+p(His); P(sigma=0) = rest.
                          --ens_unit neuron (independent draw per neuron) | type (one draw per cell type from the type-mean
                          probabilities, all members share it -- Dale's principle; untyped neurons drawn individually).
                          --ens_gt ignore (default) | fix (neurons with a ground_truth NT keep sign(ground_truth)).
                          Draws are made upfront from default_rng([ens_seed, 1]) so --ens_only i j ... can split draws
                          over array tasks reproducibly. --sign_override is applied on top of each draw.
                          Outputs: rho/surv/reg_rho concatenate the draws along the replica axis (B' = n_draws * B,
                          'ens_draw' gives each replica's draw index); signs_ens [n_draws, N] int8.
 3. Weights
   --wtransform linear | cap:K | sqrt | log1p | binary   applied to synapse counts w before normalisation.
   --wnorm renorm (default): c recomputed from the TRANSFORMED weights (raw: c = 1/mean_i sum_j T(w_ij); in: per neuron),
                          so the mean in-strength in model units stays 1 for every transform.
           none: c of the linear weights (the convention of al_common.exc_operator; total gain drops).
 4. Short-term plasticity (per PRESYNAPTIC neuron j; resource x_j in [0,1], output a_j x_j)
   --std U TAU            Tsodyks-Markram-style depression for every presynaptic neuron:
                              dx_j/dt = (1 - x_j)/TAU  -  U a_j x_j
                          (U = depletion rate per unit ACTIVE time; model time unit = 1/alpha = mean active-state
                          duration). Integrated exactly over each step with a_j = state during the step:
                          active: x <- x_inf + (x - x_inf) e^{-(1/TAU + U) dt}, x_inf = 1/(1 + U TAU);
                          quiescent: x <- 1 + (x - 1) e^{-dt/TAU}. Mean-field steady state x* = 1/(1 + U rho TAU).
                          TAU may be 'inf' (no recovery). x starts at 1.
   --std_group "SEL=U,TAU" (repeatable, later wins) per-class parameters for presynaptic SEL (e.g. "olfactory=0.5,20",
                          "ALLN=..", "ALPN=..", "sign:-1=0,1" to exempt inhibitory outputs).
   --std_mode rate|spike  (added 2026-11-09 by experiment E3; default rate = the behaviour above, unchanged.)
                          rate: depletion runs continuously DURING every active epoch, so even a fully recovered neuron's
                          first epoch is weakened: its expected integrated output is phi = x_inf + (1 - x_inf)(1 - e^{-dt}) /
                          (1 - e^{-(1 + k) dt}) of the undepressed one (k = 1/TAU + U, x_inf = (1/TAU)/k) -> the linear
                          (silent-state) branching operator, and g_lin, are rescaled by phi.
                          spike: Tsodyks-Markram per spike, one active epoch = one spike. U is then the FRACTION of the
                          resource used per epoch (0 <= U <= 1; U = 1 - f with f the retained fraction per spike). At every
                          0 -> 1 transition of j the epoch's efficacy e_j is set to the current resource x_j, then
                          x_j <- (1 - U) x_j; the output during the epoch is a_j e_j (constant over the epoch); x recovers
                          in every step, x <- 1 + (x - 1) e^{-dt/TAU}. A fully recovered neuron's epoch is undepressed, so the
                          silent-state branching operator and g_lin are EXACTLY unchanged. Neurons active at t = 0 (decay
                          start, clamp_on) count as an onset at t = 0 (e = 1, x = 1 - U). A clamp_on neuron has no later onset
                          and so never depresses; neurons switched on by a phase-2 clamp keep their last epoch's efficacy.
                          Stationary resource at the onsets of an isolated unit (renewal, epochs of L >= 1 and gaps of Q >= 1
                          steps, M = E[e^{-L dt/TAU}] E[e^{-Q dt/TAU}]): E[e] = (1 - M) / (1 - (1 - U) M); Poisson mean-field
                          x* = 1/(1 + U nu TAU), nu = epoch rate = alpha_eff rho. res_x_act records the mean EFFICACY e
                          of active depressing neurons in this mode (res_x is still the resource x).
   --adapt D TAUA         spike-frequency adaptation: dtheta_i/dt = -theta_i/TAUA + D a_i (subtracted from s_i);
                          exact per-step update theta <- theta e^{-dt/TAUA} + D TAUA (1 - e^{-dt/TAUA}) a_i;
                          stationary E[theta] = D TAUA rho exactly.
   --adapt_group "SEL=D,TAUA" (repeatable) per-population parameters.
   --refractory N         absolute refractory period of N steps after every active -> quiescent transition.
 5. Outputs (--out x.npz; plus the JSON sidecar x.json with every option and derived numbers)
   t, g, rho[g,t,B'], surv[g,t,B'], occ[g,N] (float16; swc.py conventions: replica-mean activity per neuron over the
   second-half steps actually run -- after an early stop it is divided by the steps before absorption, as in swc.py),
   reg_rho[g,t,R,B'] (fraction active in each region), reg_names, reg_sizes (default regions AL_module, AL_rest,
   CNS_rest -- disjoint, cover all N; --no_default_regions drops them; more, possibly overlapping, with
   --region "NAME=SEL" (repeatable)), res_x[d,g,t,R] (mean resource over depressing neurons of the region,
   replica mean), res_x_act[d,g,t,R] (mean resource over ACTIVE depressing neurons), res_theta[d,g,t,R] (mean theta
   over adapting neurons); NaN after every replica is absorbed (early stop) or where undefined; sec_per_step[d,g];
   signs_ens / ens_draw with --sign_ensemble; meta (JSON of the options).

 6. Engine (added 2026-11-09 by experiment E2; default unchanged)
   --engine spmm (default) | push   push = event-driven product: instead of the full sparse-dense product W @ X it sums
                          the out-edges of the ACTIVE (neuron, replica) pairs only (index_add_ over a pre-sorted copy of
                          W), which is ~10x faster while activity is sparse (localized phases). Each step falls back to
                          torch.sparse.mm when the active out-edge count exceeds --push_max * nnz * B. Results equal the
                          spmm engine up to float32 summation order (atomics), so trajectories are statistically, not
                          bitwise, identical; RNG calls are unchanged. push always uses the general code path.
   --push_max F           fallback threshold (default 0.05).
 7. Two-phase protocols and extra outputs (added 2026-11-09 for experiment E1, ORN drive / antenna removal; all OFF by
    default, so the default path and its outputs are unchanged)
   --switch_at T1         after the step that ends at time T1 (T1/dt rounded to a step) switch to phase-2 settings:
   --sw_drive "SEL=H"     (repeatable) phase-2 drive overrides, applied on top of the phase-1 drive vector;
   --sw_clamp_off "SEL"   (repeatable) phase-2 extra clamps to 0 (e.g. antenna removal: "olfactory");
   --sw_clamp_on "SEL"    (repeatable) phase-2 extra clamps to 1.
                          The new clamps act on the state immediately at T1 (the record AT T1 is the last phase-1 state).
                          Silence becomes absorbing (early stop) if phase 2 has no positive drive and no clamp_on.
                          Depression / adaptation / refractory state carries over unchanged.
   --rec_lin DT           also record rho/surv/reg_rho every DT time units (union with the default log grid).
                          --switch_at always adds the record at T1 and a 150-point log grid in time since the switch.
   --win "T0:T1"          (repeatable) per-replica window averages of every region's active fraction over all steps with
                          T0 < t <= T1: win_mean[g, W, R, B] and win_m2[g, W, R, B] (mean of the square), divided by the
                          FULL number of steps in the window (steps skipped by an early stop count as silent).
   --reg_last             reg_last[g, R, B]: the last time at which the region had >= 1 active neuron (-1 = never).
   --occ_win "T0:T1"      per-neuron occupancy occ over T0 < t <= T1 (full-window denominator) instead of the second half.
   The region outputs need regions (the default three or --region).
 8. Model-2 additions (2026-11-09, results/m2/MODEL2.md; all OFF by default, so every earlier path and output is unchanged)
   --edge_sign "PRE->POST=V"  (repeatable, later wins; applied after --sign_override and on top of every ensemble draw)
                          per-EDGE sign: every edge whose presynaptic neuron is in selector PRE and whose postsynaptic
                          neuron is in selector POST gets sigma_e = V (V >= 0 as is, V < 0 scaled by r as in
                          --sign_override); e.g. "lLN1_bc->ALPN=0" --edge_sign "lLN1_bc->ALLN=1" (model M2s). Edges whose
                          coefficient becomes 0 are dropped. The sidecar lists edges / synapses changed per override.
                          Depression (a property of the presynaptic neuron) is untouched by it.
   --null_seed S          seed of the null-graph permutation (--null); default: the run's own RNG as in swc.py (the null
                          graph then changes with --seed). With S fixed, runs with different --seed share one null graph.
   --graph FILE.npz       run on another graph instead of the cache: pre, post (local ids 0..N-1), w (synapse counts),
                          sign (per local neuron, +1/-1/0); optional sub (global cache ids of the local neurons: then every
                          label selector works on the subgraph) and c (raw normalisation constant; default 1/mean
                          in-strength of this graph). Written by code/m2_common.py write_graph() (pd3 FSS graphs).
   --rec_counts           per-step counts for every replica: cnt_act[g, s, B] (active units after step s) and
                          cnt_on[g, s, B] (0 -> 1 transitions during step s, pulse seeds included), int32, for the steps
                          s with T0 < s*dt <= T1 of --counts_win "T0:T1" (default the whole run); cnt_t = times.
                          --counts_reg adds cnt_reg_act / cnt_reg_on [g, s, R, B] for every region. Enough to extract
                          avalanches (code/m2_obs.py) and to compute spectra / autocorrelations of rho(t) offline.
   --pulse "SEL"          separation-of-timescales drive: a replica that is completely silent at the end of a step gets
                          ONE uniformly chosen neuron of SEL (minus clamped neurons) switched on during the next step, with
                          probability 1 - exp(-dt/G) per silent step (--pulse_gap G, default 0 = at once). The silent
                          interval between avalanches is then Q steps, Q ~ Geometric(1 - e^{-dt/G}) on {1, 2, ...} (at
                          least one silent step separates avalanches). The seed is an ordinary onset (depresses its
                          resource, counts in cnt_on); a neuron in its refractory period is not seeded (retry next step).
                          Pulse randomness comes from a separate generator (seed --seed), so the main stream is untouched.
                          Silence is no longer absorbing (no early stop).
   --ramp D               quasi-static g schedule IN ONE RUN: the --g values are visited in the order given, each for
                          D time units, carrying the state, resources, thresholds and refractory clocks over;
                          --ramp_loop appends the reversed list (without repeating the turning point): up and back down.
                          --T is replaced by n_seg * D. Outputs (dimension g of rho/surv/... is 1): seg_g[n_seg],
                          seg_rho / seg_rho2 [n_seg, B] (mean and mean square of the active fraction over the last
                          --ramp_avg fraction (default 0.5) of each dwell), seg_alive[n_seg, B] (any activity at the dwell's
                          end), with regions seg_reg / seg_reg2 [n_seg, R, B], with depression seg_x / seg_xa [n_seg, R]
                          (mean resource / efficacy of active depressing neurons at the dwell's end; NaN if undefined).
                          Records are added at every dwell end. An early stop (all silent, absorbing) leaves the remaining
                          segments at 0 (seg_x NaN).
   --init "SEL"           (repeatable, union; needs --exp quiet) start with SEL active and everything else silent, e.g. one
                          AL lobe ("module&side:L"); --init_frac F activates each SEL neuron independently with
                          probability F in every replica (separate generator). Active-at-start depressing neurons count as
                          an onset at t = 0 (as --exp decay).

 Other flags as swc.py: --exp decay|spread|quiet (quiet = start all quiescent, for driven runs), --norm in|raw, --g,
 --r, --beta, --T, --dt, --B, --null [--null_kind targets|weights], --seed_set, --seed; plus --device, --no_occ.
 Early stop (as swc.py): when no neuron has positive drive and all replicas are silent (clamp_on counts as drive).

 Usage: python ald_sim.py --exp decay --norm raw --g 0.3 0.5 --r 1 --T 3000 --B 64 --out results/aldef/x.npz \
          --drive "olfactory=0.01" --std 0.5 20 --sign_override "lLN1_bc=-1"
"""
import argparse, json, os, re, time
import numpy as np
import torch

C = "/MALECNS_CACHE"
RAW_NT = "/MALECNS_RAW/body-neurotransmitters-male-cns-v1.0.feather"
ROOT = "/MALECNS_ROOT"
MODULE_NPZ = f"{ROOT}/results/module/module_graph.npz"
NT_CACHE = f"{ROOT}/results/aldef/nt_table.npz"
AL_CLASSES = ("olfactory", "hygrosensory", "thermosensory", "ALPN", "ALLN", "ALIN", "ALON")   # = al_common.AL_CLASSES
NT_CLASSES = ("acetylcholine", "gaba", "glutamate", "histamine", "dopamine", "octopamine", "serotonin", "unclear")
NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}               # everything else 0


# ---------------------------------------------------------------------------------------------------------------- args
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", choices=["decay", "spread", "quiet"], required=True)
    ap.add_argument("--norm", choices=["in", "raw"], default="in")
    ap.add_argument("--g", type=float, nargs="+", required=True)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=2000.0)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--null_kind", choices=["targets", "weights"], default="targets")
    ap.add_argument("--seed_set", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    # new
    ap.add_argument("--drive", action="append", default=[], metavar="SEL=H")
    ap.add_argument("--clamp_off", action="append", default=[], metavar="SEL")
    ap.add_argument("--clamp_on", action="append", default=[], metavar="SEL")
    ap.add_argument("--sign_override", action="append", default=[], metavar="SEL=V")
    ap.add_argument("--sign_ensemble", type=int, default=0, metavar="K")
    ap.add_argument("--ens_unit", choices=["neuron", "type"], default="neuron")
    ap.add_argument("--ens_conf", choices=["body", "celltype"], default="body")
    ap.add_argument("--ens_gt", choices=["ignore", "fix"], default="ignore")
    ap.add_argument("--ens_qmiss", type=float, default=0.5)
    ap.add_argument("--ens_seed", type=int, default=None)
    ap.add_argument("--ens_only", type=int, nargs="+", default=None)
    ap.add_argument("--wtransform", default="linear")
    ap.add_argument("--wnorm", choices=["renorm", "none"], default="renorm")
    ap.add_argument("--std", type=float, nargs=2, default=None, metavar=("U", "TAU"))
    ap.add_argument("--std_group", action="append", default=[], metavar="SEL=U,TAU")
    ap.add_argument("--std_mode", choices=["rate", "spike"], default="rate")          # E3 addition (default unchanged)
    ap.add_argument("--adapt", type=float, nargs=2, default=None, metavar=("D", "TAUA"))
    ap.add_argument("--adapt_group", action="append", default=[], metavar="SEL=D,TAUA")
    ap.add_argument("--refractory", type=int, default=0)
    ap.add_argument("--region", action="append", default=[], metavar="NAME=SEL")
    ap.add_argument("--no_default_regions", action="store_true")
    ap.add_argument("--no_occ", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--engine", choices=["spmm", "push"], default="spmm")
    ap.add_argument("--push_max", type=float, default=0.05)
    # E1 additions (default off)
    ap.add_argument("--switch_at", type=float, default=None, metavar="T1")
    ap.add_argument("--sw_drive", action="append", default=[], metavar="SEL=H")
    ap.add_argument("--sw_clamp_off", action="append", default=[], metavar="SEL")
    ap.add_argument("--sw_clamp_on", action="append", default=[], metavar="SEL")
    ap.add_argument("--rec_lin", type=float, default=0.0, metavar="DT")
    ap.add_argument("--win", action="append", default=[], metavar="T0:T1")
    ap.add_argument("--reg_last", action="store_true")
    ap.add_argument("--occ_win", default=None, metavar="T0:T1")
    # Model-2 additions (default off)
    ap.add_argument("--edge_sign", action="append", default=[], metavar="PRE->POST=V")
    ap.add_argument("--null_seed", type=int, default=None)
    ap.add_argument("--graph", default=None, metavar="FILE.npz")
    ap.add_argument("--rec_counts", action="store_true")
    ap.add_argument("--counts_win", default=None, metavar="T0:T1")
    ap.add_argument("--counts_reg", action="store_true")
    ap.add_argument("--pulse", default=None, metavar="SEL")
    ap.add_argument("--pulse_gap", type=float, default=0.0, metavar="G")
    ap.add_argument("--ramp", type=float, default=0.0, metavar="D")
    ap.add_argument("--ramp_loop", action="store_true")
    ap.add_argument("--ramp_avg", type=float, default=0.5)
    ap.add_argument("--init", action="append", default=[], metavar="SEL")
    ap.add_argument("--init_frac", type=float, default=1.0, metavar="F")
    return ap


# ------------------------------------------------------------------------------------------------------------ labels
class Labels:
    """Lazy per-neuron annotation arrays of the cache (or of a small test graph: pass a dict of arrays)."""
    KEYS = {"type": "types", "class": "classes", "superclass": "superclasses", "subclass": "subclasses",
            "instance": "instances", "nt": "nt", "side": "sides"}

    def __init__(self, N, arrays=None, sign=None, module_idx=None, sub=None):
        self.N, self._a, self.sign = N, dict(arrays or {}), sign
        self._module = module_idx
        self.from_cache = arrays is None
        self.sub = None if sub is None else np.asarray(sub, np.int64)      # --graph: global ids of the local neurons

    def get(self, key):
        f = self.KEYS[key]
        if f not in self._a:
            if not self.from_cache:
                return np.full(self.N, "", dtype=object).astype(str)
            v = np.load(f"{C}/{f}.npy", allow_pickle=True).astype(str)
            self._a[f] = v if self.sub is None else v[self.sub]
        return self._a[f]

    def module(self):
        if self._module is None:
            self._module = np.load(MODULE_NPZ, allow_pickle=True)["idx"]
        if self.sub is not None:
            return np.isin(self.sub, self._module)
        m = np.zeros(self.N, bool); m[self._module] = True
        return m

    def al(self):
        return np.isin(self.get("class"), AL_CLASSES)


def _re_mask(arr, pattern):
    u, inv = np.unique(arr, return_inverse=True)
    rx = re.compile(pattern)
    hit = np.array([rx.fullmatch(x) is not None for x in u], bool)
    return hit[inv]


def _bare_mask(arr, tok):
    m = arr == tok
    try:
        m = m | _re_mask(arr, tok)
    except re.error:
        pass
    return m


def _atom(tok, L):
    if tok == "all":
        return np.ones(L.N, bool)
    if tok == "none":
        return np.zeros(L.N, bool)
    if tok == "module":
        return L.module()
    if tok == "al":
        return L.al()
    if tok == "al_rest":
        return L.al() & ~L.module()
    if tok == "cns_rest":
        return ~(L.al() | L.module())
    if tok.startswith("idx:") or tok.endswith(".npz") or ".npz:" in tok:
        spec = tok[4:] if tok.startswith("idx:") else tok
        path, key = (spec.rsplit(":", 1) if ".npz:" in spec else (spec, None))
        d = np.load(path, allow_pickle=True)
        v = d[key] if key else (d["idx"] if "idx" in d.files else d["mask"])
        if L.sub is not None:           # --graph with sub: index files hold GLOBAL cache ids (or a full-size mask)
            if v.dtype == bool:
                return v.astype(bool)[L.sub] if len(v) != L.N else v.astype(bool).copy()
            return np.isin(L.sub, np.asarray(v, np.int64))
        if v.dtype == bool:
            return v.astype(bool).copy()
        m = np.zeros(L.N, bool); m[np.asarray(v, np.int64)] = True
        return m
    if ":" in tok:
        k, v = tok.split(":", 1)
        if k == "sign":
            return L.sign == float(v)
        if k in Labels.KEYS:
            return _re_mask(L.get(k), v)
        raise ValueError(f"unknown selector key {k!r} in {tok!r}")
    if tok == "":
        raise ValueError("empty selector token")
    return _bare_mask(L.get("type"), tok) | _bare_mask(L.get("class"), tok) | _bare_mask(L.get("superclass"), tok)


def select(expr, L, allow_empty=False):
    """Boolean mask for a selector expression (grammar in the module docstring)."""
    mask = np.zeros(L.N, bool)
    for clause in expr.split(";"):
        m = np.ones(L.N, bool)
        for term in clause.split("&"):
            term = term.strip(); neg = term.startswith("~")
            t = _atom(term[1:].strip() if neg else term, L)
            m &= ~t if neg else t
        mask |= m
    if not allow_empty and not mask.any():
        raise ValueError(f"selector {expr!r} selects no neuron")
    return mask


def split_assign(s):
    """'SEL=VALUE' -> (SEL, VALUE) splitting at the LAST '='."""
    if "=" not in s:
        raise ValueError(f"expected SEL=VALUE, got {s!r}")
    a, b = s.rsplit("=", 1)
    return a.strip(), b.strip()


# --------------------------------------------------------------------------------------------------------- NT / signs
def nt_table(body_ids=None):
    """Per-neuron NT fields of the release joined on bodyId (cached as npz; built atomically on first use)."""
    if os.path.exists(NT_CACHE):
        d = np.load(NT_CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    import pyarrow.feather as feather
    df = feather.read_table(RAW_NT).to_pandas().set_index("body")
    bid = np.load(f"{C}/body_ids.npy") if body_ids is None else body_ids
    d = df.reindex(bid)
    out = {"consensus": d["consensus_nt"].fillna("unknown").to_numpy().astype(str),
           "pred": d["predicted_nt"].fillna("unknown").to_numpy().astype(str),
           "pconf": d["predicted_nt_confidence"].to_numpy().astype(np.float64),
           "ctpred": d["celltype_predicted_nt"].fillna("unknown").to_numpy().astype(str),
           "ctconf": d["celltype_predicted_nt_confidence"].to_numpy().astype(np.float64),
           "gt": d["ground_truth"].fillna("").to_numpy().astype(str)}
    os.makedirs(os.path.dirname(NT_CACHE), exist_ok=True)
    tmp = NT_CACHE + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **out); os.replace(tmp, NT_CACHE)
    return out


def sign_probs(tab, conf="body", gt="ignore", qmiss=0.5):
    """P[N, 3] over model signs (+1, -1, 0) from the release's NT predictions (mapping in the module docstring).
    Returns P and q (the confidence assigned to the consensus label)."""
    L = tab["consensus"]
    tiers = [("pred", "pconf"), ("ctpred", "ctconf")]
    if conf == "celltype":
        tiers = tiers[::-1]
    q = np.full(len(L), np.nan)
    for pk, ck in tiers:
        m = np.isnan(q) & (tab[pk] == L) & np.isfinite(tab[ck])
        q[m] = tab[ck][m]
    q[np.isnan(q)] = qmiss
    cnt = np.array([(L == n).sum() for n in NT_CLASSES], float)
    pi = cnt / cnt.sum()                                # release-wide consensus frequencies (prior over alternatives)
    sgn = np.array([NT_SIGN.get(n, 0) for n in NT_CLASSES])
    P = _sign_probs_exact(L, q, pi, sgn)                # columns: +1, -1, 0
    col = {1: 0, -1: 1, 0: 2}
    if gt == "fix":
        g = tab["gt"]; has = g != ""
        s = np.array([NT_SIGN.get(x, 0) for x in g[has]])
        P[has] = 0.0
        P[np.where(has)[0], np.array([col[int(v)] for v in s], int)] = 1.0
    return P, q


def _sign_probs_exact(L, q, pi, sgn):
    """p(nt) = q [nt = L] + (1 - q) pi(nt)/(1 - pi(L)) for nt != L (L = 'unknown': pi(L) = 0, own sign 0)."""
    col = {1: 0, -1: 1, 0: 2}
    kL = np.full(len(L), -1)
    for k, n in enumerate(NT_CLASSES):
        kL[L == n] = k
    piL = np.where(kL >= 0, pi[np.maximum(kL, 0)], 0.0)
    P = np.zeros((len(L), 3))
    for k in range(len(NT_CLASSES)):
        alt = np.where(kL == k, 0.0, (1 - q) * pi[k] / (1 - piL))
        P[:, col[int(sgn[k])]] += alt
    own = np.where(kL >= 0, np.array([col[int(s)] for s in sgn])[np.maximum(kL, 0)], 2)   # unknown -> sign 0
    P[np.arange(len(L)), own] += q
    return P


def draw_signs(P, K, rng, unit="neuron", types=None):
    """K draws of sign vectors (int8 [K, N]) from P[N, 3]."""
    vals = np.array([1, -1, 0], np.int8)
    N = len(P)
    cum = np.cumsum(P, 1); cum[:, -1] = 1.0
    out = np.empty((K, N), np.int8)
    if unit == "neuron" or types is None:
        for k in range(K):
            u = rng.random(N)
            out[k] = vals[(u[:, None] > cum).sum(1)]
        return out
    ut, inv = np.unique(types, return_inverse=True)
    Pt = np.zeros((len(ut), 3))
    np.add.at(Pt, inv, P); Pt /= np.bincount(inv, minlength=len(ut))[:, None]
    ct = np.cumsum(Pt, 1); ct[:, -1] = 1.0
    untyped = types == ""
    for k in range(K):
        ut_draw = vals[(rng.random(len(ut))[:, None] > ct).sum(1)]
        s = ut_draw[inv]
        u = rng.random(N)
        s[untyped] = vals[(u[untyped, None] > cum[untyped]).sum(1)]
        out[k] = s
    return out


def sigma_vector(sign, r, overrides=()):
    """Per-neuron output coefficient sigma (float64) with swc.py's values, then overrides [(mask, V)]."""
    sig = np.where(sign > 0, 1.0, np.where(sign < 0, -r, 0.0))
    for m, v in overrides:
        sig[m] = v if v >= 0 else v * r
    return sig


# ------------------------------------------------------------------------------------------------------- weights
def wtransform(w, spec):
    """Synapse-count transform (float32 in, float32 out)."""
    if spec == "linear":
        return w
    if spec.startswith("cap:"):
        return np.minimum(w, np.float32(float(spec[4:]))).astype(np.float32)
    if spec == "sqrt":
        return np.sqrt(w).astype(np.float32)
    if spec == "log1p":
        return np.log1p(w).astype(np.float32)
    if spec == "binary":
        return (w > 0).astype(np.float32)
    raise ValueError(f"unknown --wtransform {spec!r}")


def node_scale(post, w_lin, wT, N, norm, wnorm):
    """Per-neuron scale c (float64 array) exactly as swc.py for the linear case."""
    wref = wT if wnorm == "renorm" else w_lin
    ins = np.bincount(post, weights=wref, minlength=N).astype(np.float32)
    if norm == "in":
        return np.where(ins > 0, 1.0 / np.maximum(ins, 1e-9), 0.0)
    return np.full(N, 1.0 / ins[ins > 0].mean())


def build_matrix(pre, post, wT, scale, sig, N, dev, edge_ov=None):
    """CSR operator W[post, pre] = sigma_pre * T(w) * c_post, built with swc.py's exact float32 arithmetic.
    edge_ov: [(edge mask, sigma value), ...] per-edge sign overrides (--edge_sign; later entries win), or None."""
    sp = sig[pre].astype(np.float32)
    for em, v in (edge_ov or ()):
        sp[em] = np.float32(v)
    coef = sp * wT * scale[post].astype(np.float32)
    keep = coef != 0
    idx = torch.tensor(np.vstack([post[keep], pre[keep]]), device=dev)
    Wm = torch.sparse_coo_tensor(idx, torch.tensor(coef[keep], device=dev), (N, N)).coalesce().to_sparse_csr()
    return Wm, int(keep.sum())


class PushMV:
    """Event-driven product W @ Y (--engine push): sums the out-edges of the nonzero (neuron, replica) entries of Y.
    W is the CSR operator [post, pre] of build_matrix; a copy sorted by presynaptic neuron is kept on the device.
    Falls back to torch.sparse.mm when the active out-edge count E > push_max * nnz * B (dense activity)."""

    def __init__(self, Wm, N, push_max=0.05):
        coo = Wm.to_sparse_coo().coalesce()
        post, pre = coo.indices()
        val = coo.values()
        pre_s, o = torch.sort(pre, stable=True)
        self.tgt, self.val = post[o].contiguous(), val[o].contiguous()
        self.deg = torch.bincount(pre_s, minlength=N)
        self.ptr = torch.cumsum(self.deg, 0) - self.deg                      # start of each neuron's out-edges
        self.nnz, self.N, self.push_max = int(val.numel()), N, float(push_max)
        self.stats = {"push": 0, "spmm": 0, "zero": 0}
        del coo, post, pre, val, pre_s, o

    def __call__(self, Wm, Y, binary=True):
        N, B = Y.shape
        r_, c_ = Y.nonzero(as_tuple=True)
        m = r_.numel()
        if m == 0:
            self.stats["zero"] += 1
            return torch.zeros_like(Y)
        deg = self.deg[r_]
        E = int(deg.sum())
        if E > self.push_max * self.nnz * B:
            self.stats["spmm"] += 1
            return torch.sparse.mm(Wm, Y)
        self.stats["push"] += 1
        cs = torch.cumsum(deg, 0)
        rep = torch.repeat_interleave(torch.arange(m, device=Y.device), deg, output_size=E)
        pos = torch.arange(E, device=Y.device) + (self.ptr[r_] - (cs - deg))[rep]
        v = self.val[pos]
        if not binary:
            v = v * Y[r_, c_][rep]
        out = torch.zeros(N * B, device=Y.device, dtype=Y.dtype)
        out.index_add_(0, self.tgt[pos] * B + c_[rep], v)
        return out.view(N, B)


# ------------------------------------------------------------------------------------------------------ dynamics
def make_params(N, dt, dev, std_spec=None, adapt_spec=None, nref=0, h=0.0, hv=None, off=None, on=None, std_mode="rate"):
    """Per-neuron parameter tensors. std_spec/adapt_spec: (U[N], TAU[N]) / (D[N], TAUA[N]) numpy arrays or None.
    hv: per-neuron drive (numpy) or None (scalar h); off/on: bool masks or None.
    std_mode: "rate" (default; continuous depletion while active) or "spike" (per-epoch Tsodyks-Markram; E3 addition)."""
    P = {"dt": dt, "h": h, "nref": int(nref), "hv": None, "keep": None, "on": None, "std": None, "adapt": None}
    if hv is not None:
        P["hv"] = torch.tensor(hv, dtype=torch.float32, device=dev)[:, None]
    if off is not None and off.any():
        P["keep"] = torch.tensor(~off, dtype=torch.float32, device=dev)[:, None]
    if on is not None and on.any():
        P["on"] = torch.tensor(on, dtype=torch.float32, device=dev)[:, None]
    if std_spec is not None:
        U, TAU = (np.asarray(v, np.float64) for v in std_spec)
        rinv = np.where(np.isfinite(TAU), 1.0 / TAU, 0.0)
        k = rinv + U
        xinf = np.where(k > 0, rinv / np.where(k > 0, k, 1.0), 1.0)
        ea, eq = np.exp(-k * dt), np.exp(-rinv * dt)
        f = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)[:, None]
        P["std"] = {"xinf": f(xinf), "ea": f(ea), "eq": f(eq), "on": f(U > 0), "n_on": int((U > 0).sum())}
        if std_mode == "spike":
            if np.any(U > 1):
                raise ValueError("--std_mode spike: U is the fraction used per epoch and must be <= 1")
            P["std"].update({"mode": "spike", "keepf": f(1.0 - U)})
        elif std_mode != "rate":
            raise ValueError(f"unknown std_mode {std_mode!r}")
    if adapt_spec is not None:
        D, TA = (np.asarray(v, np.float64) for v in adapt_spec)
        e = np.exp(-dt / TA)
        f = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)[:, None]
        P["adapt"] = {"e": f(e), "inc": f(D * TA * (1 - e)), "on": f(D > 0), "n_on": int((D > 0).sum())}
    return P


def simulate(Wm, N, gs, P, *, exp, T, dt, B, beta, rng, dev, seed_set=None, regions=None, occ=True, log=None,
             push=None, switch=None, rec_extra=None, windows=None, occ_steps=None, reg_last=False,
             counts=None, pulse=None, ramp=None, init=None):
    """Run every g in gs with the parameters P (make_params). Returns a dict of numpy arrays.
    Model-2 options (all None = unchanged behaviour and unchanged main RNG stream):
      counts    {"s0", "s1", "reg"}: per-step int32 counts of active units and of onsets for steps s0 < s <= s1
                (per region too if reg); keys cnt_act, cnt_on [g, s, B], cnt_reg_act, cnt_reg_on [g, s, R, B], cnt_t.
      pulse     {"idx": LongTensor of seedable neurons, "p": seed probability per silent step, "gen": torch.Generator}.
      ramp      {"g": schedule (one g per dwell), "L": steps per dwell, "A": averaged steps at each dwell's end}; gs must
                then hold one entry (the run is one trajectory through the schedule).
      init      {"mask": bool tensor [N, 1], "frac": F, "gen": torch.Generator}: quiet start with the masked neurons active
                (each independently with probability F per replica when F < 1; separate generator).
    With P all-default this is line-for-line the swc.py loop (same tensor ops, same RNG calls).
    E1 options (all None/False = unchanged behaviour; none of them draws random numbers):
      switch    {"s": step index s_sw, "P": phase-2 params (make_params)}: steps s > s_sw use P2's h/hv/keep/on; P2's
                clamps are applied to the state right after step s_sw.
      rec_extra extra step indices to record (union with the default log grid).
      windows   [(s0, s1), ...] step ranges: per-replica mean and mean square of each region's active fraction over
                s0 < s <= s1, divided by s1 - s0 (requires regions).
      occ_steps (s0, s1): per-neuron occupancy over s0 < s <= s1, divided by s1 - s0 (instead of the second half).
      reg_last  record the last step time at which each region had an active neuron (requires regions)."""
    steps = int(round(T / dt))
    rec = np.unique(np.round(np.logspace(0, np.log10(steps), 120)).astype(int))
    if rec_extra is not None:
        rec = np.unique(np.concatenate([rec, np.asarray(rec_extra, int)]))
        rec = rec[(rec >= 1) & (rec <= steps)]
    p_off = 1 - np.exp(-1.0 * dt)
    h, hv, keep, on, std, ad, nref = P["h"], P["hv"], P["keep"], P["on"], P["std"], P["adapt"], P["nref"]
    # silence is absorbing without positive drive; for a scalar h use swc.py's exact test (h == 0) so outputs match
    nofeat = counts is None and pulse is None and ramp is None          # Model-2 options all off
    absorbing = on is None and (h == 0 if hv is None else float(hv.max()) <= 0) and pulse is None
    plain = (hv is None and keep is None and on is None and std is None and ad is None and nref == 0
             and push is None and nofeat)
    if ramp is not None and len(gs) != 1:
        raise ValueError("ramp: simulate() runs one trajectory (gs must hold one entry)")
    if switch is not None:
        P2 = switch["P"]
        absorbing2 = (P2["on"] is None and (P2["h"] == 0 if P2["hv"] is None else float(P2["hv"].max()) <= 0)
                      and pulse is None)
    windows = list(windows or [])
    per_step_reg = bool(windows) or reg_last
    if per_step_reg and regions is None:
        raise ValueError("--win / --reg_last need regions")
    if regions is not None:
        Rm = regions["Rm"]; Rsz = Rm.sum(1)
        if std is not None:
            Rx = Rm * std["on"][:, 0][None]; Rxs = Rx.sum(1)
        if ad is not None:
            Ra = Rm * ad["on"][:, 0][None]; Ras = Ra.sum(1)
    out = {"rho": [], "surv": [], "occ": [], "reg": [], "x": [], "xa": [], "th": [], "sps": [],
           "wm": [], "w2": [], "last": [], "ca": [], "co": [], "cra": [], "cro": [], "seg": []}
    nanR = np.full(len(regions["names"]) if regions else 0, np.nan)
    if counts is not None:
        c0, c1 = int(counts["s0"]), min(int(counts["s1"]), steps)
        if counts.get("reg") and regions is None:
            raise ValueError("--counts_reg needs regions")
    if ramp is not None:
        rL, rA, rG = int(ramp["L"]), int(ramp["A"]), np.asarray(ramp["g"], float)
        if rL * len(rG) != steps or not 1 <= rA <= rL:
            raise ValueError("ramp: steps must equal n_seg * L and 1 <= A <= L")
    if pulse is not None:
        pg, pidx, pp = pulse["gen"], pulse["idx"], float(pulse["p"])
        ar = torch.arange(B, device=dev)
    for g in gs:
        tg = time.time()
        if switch is not None:          # phase-1 settings for every g
            h, hv, keep, on = P["h"], P["hv"], P["keep"], P["on"]
            # no early stop in phase 1 if phase 2 is driven (silence before the switch is then not final)
            absorbing = absorbing2 and on is None and (h == 0 if hv is None else float(hv.max()) <= 0)
            plain = (hv is None and keep is None and on is None and std is None and ad is None and nref == 0
                     and push is None and nofeat)
        if counts is not None:
            c_act = torch.zeros(max(c1 - c0, 0), B, device=dev, dtype=torch.int32); c_on = torch.zeros_like(c_act)
            if counts.get("reg"):
                c_ract = torch.zeros(max(c1 - c0, 0), Rm.shape[0], B, device=dev, dtype=torch.int32)
                c_ron = torch.zeros_like(c_ract)
        if ramp is not None:
            nS = len(rG); f64 = dict(device=dev, dtype=torch.float64)
            sg_s, sg_q, sg_al = torch.zeros(nS, B, **f64), torch.zeros(nS, B, **f64), torch.zeros(nS, B, **f64)
            if regions is not None:
                sg_rs = torch.zeros(nS, Rm.shape[0], B, **f64); sg_rq = torch.zeros_like(sg_rs)
                sg_x = torch.full((nS, Rm.shape[0]), float("nan"), **f64); sg_xa = torch.full_like(sg_x, float("nan"))
        if per_step_reg:
            nR = Rm.shape[0]
            wsum = torch.zeros(max(len(windows), 1), nR, B, device=dev, dtype=torch.float64)
            wsq = torch.zeros_like(wsum)
            last = torch.full((nR, B), -1.0, device=dev, dtype=torch.float64)
        if exp == "decay":
            X = torch.ones(N, B, device=dev)
        else:
            X = torch.zeros(N, B, device=dev)
            if init is not None and exp == "quiet":       # Model-2 addition: partial initial state
                if init["frac"] >= 1:
                    X = X + init["mask"].float().expand_as(X)
                else:
                    X = ((torch.rand(N, B, generator=init["gen"], device=dev) < init["frac"]) & init["mask"]).float()
            if exp == "spread":
                if seed_set:
                    ss = np.load(seed_set)
                    seeds = rng.choice(ss["idx"], B, replace=True, p=ss["p"])
                else:
                    seeds = rng.choice(N, B, replace=False)
                X[torch.tensor(seeds, device=dev), torch.arange(B, device=dev)] = 1
        if keep is not None:
            X = X * keep
        if on is not None:
            X = torch.maximum(X, on.expand_as(X))
        xr = torch.ones(N, B, device=dev) if std is not None else None
        spike = std is not None and std.get("mode") == "spike"
        ef = None
        if spike:                       # neurons active at t = 0 count as an onset at t = 0: e = 1, x = 1 - U
            ef = torch.ones(N, B, device=dev)
            xr = torch.where(X > 0, std["keepf"].expand_as(X), xr)
        th = torch.zeros(N, B, device=dev) if ad is not None else None
        Rf = torch.zeros(N, B, device=dev) if nref > 0 else None
        rho, surv, regt, xt, xat, tht, k = [], [], [], [], [], [], 0
        oc = torch.zeros(N, device=dev); n_occ = 0
        s_done = 0
        for s in range(1, steps + 1):
            if ramp is not None and (s - 1) % rL == 0:          # next dwell of the g schedule
                g = float(rG[(s - 1) // rL])
            if switch is not None and s == switch["s"] + 1:      # phase 2 starts: new drive / clamps
                h, hv, keep, on = P2["h"], P2["hv"], P2["keep"], P2["on"]
                absorbing, plain = absorbing2, False
                if keep is not None:
                    X = X * keep
                if on is not None:
                    X = torch.maximum(X, on.expand_as(X))
            if plain:
                inp = g * torch.sparse.mm(Wm, X) + h
                p_on = 1 - torch.exp(-beta * torch.tanh(inp.clamp(min=0)) * dt)
                u = torch.rand_like(X)
                X = torch.where(X > 0, (u >= p_off).float(), (u < p_on).float())
            else:
                Y = (X * (ef if spike else xr)) if std is not None else X
                WY = torch.sparse.mm(Wm, Y) if push is None else push(Wm, Y, std is None)
                inp = g * WY + (hv if hv is not None else h)
                if ad is not None:
                    inp = inp - th
                p_on = 1 - torch.exp(-beta * torch.tanh(inp.clamp(min=0)) * dt)
                u = torch.rand_like(X)
                act = u < p_on
                if Rf is not None:
                    act = act & (Rf == 0)
                Xn = torch.where(X > 0, (u >= p_off).float(), act.float())
                if keep is not None:
                    Xn = Xn * keep
                if on is not None:
                    Xn = torch.maximum(Xn, on.expand_as(Xn))
                if pulse is not None:    # separation of timescales: seed one SEL neuron in replicas silent after s-1
                    coin = torch.rand(B, generator=pg, device=dev) < pp
                    pick = pidx[torch.randint(len(pidx), (B,), generator=pg, device=dev)]
                    do = (X.sum(0) == 0) & coin
                    if keep is not None:                 # never seed a clamped-off neuron (also phase-2 clamps)
                        do = do & (keep[pick, 0] > 0)
                    if Rf is not None:
                        do = do & (Rf[pick, ar] == 0)
                    Xn[pick, ar] = torch.where(do, torch.ones_like(do, dtype=Xn.dtype), Xn[pick, ar])
                if counts is not None or spike:
                    ons = (X == 0) & (Xn > 0)
                if counts is not None and c0 < s <= c1:
                    ci = s - c0 - 1
                    c_act[ci] = Xn.sum(0).to(torch.int32); c_on[ci] = ons.sum(0).to(torch.int32)
                    if counts.get("reg"):
                        c_ract[ci] = (Rm @ Xn).to(torch.int32); c_ron[ci] = (Rm @ ons.float()).to(torch.int32)
                if spike:                # recovery over the step, then onsets (0 -> 1) take e = x and deplete x
                    xr = 1 + (xr - 1) * std["eq"]
                    ef = torch.where(ons, xr, ef)
                    xr = torch.where(ons, xr * std["keepf"], xr)
                elif std is not None:    # resources integrate the state held DURING the step (X, before the update)
                    xr = torch.where(X > 0, std["xinf"] + (xr - std["xinf"]) * std["ea"], 1 + (xr - 1) * std["eq"])
                if ad is not None:
                    th = th * ad["e"] + ad["inc"] * X
                if Rf is not None:
                    Rf = torch.where((X > 0) & (Xn == 0), float(nref), (Rf - 1).clamp(min=0))
                X = Xn
            s_done = s
            if ramp is not None:                     # dwell statistics (state at the end of step s)
                j, pos = divmod(s - 1, rL)
                if pos >= rL - rA:
                    fr = (X.sum(0) / N).double()
                    sg_s[j] += fr; sg_q[j] += fr * fr
                    if regions is not None:
                        rf = ((Rm @ X) / Rsz[:, None]).double()
                        sg_rs[j] += rf; sg_rq[j] += rf * rf
                if pos == rL - 1:
                    sg_al[j] = (X.sum(0) > 0).double()
                    if regions is not None and std is not None:
                        sg_x[j] = ((Rx @ xr).mean(1) / Rxs).double()
                        num, den = (Rx @ (X * (ef if spike else xr))).sum(1), (Rx @ X).sum(1)
                        sg_xa[j] = torch.where(den > 0, num / den.clamp(min=1), torch.nan).double()
            if occ_steps is not None:
                if occ and occ_steps[0] < s <= occ_steps[1]:
                    oc += X.mean(1); n_occ += 1
            elif occ and s > steps // 2:
                oc += X.mean(1); n_occ += 1
            if per_step_reg:
                cnt = Rm @ X                                   # [R, B] active counts
                if reg_last:
                    last = torch.where(cnt > 0, s * dt, last)
                if windows:
                    fr = (cnt / Rsz[:, None]).double()
                    for wi, (s0, s1) in enumerate(windows):
                        if s0 < s <= s1:
                            wsum[wi] += fr; wsq[wi] += fr * fr
            if s == rec[k]:
                n_act = X.sum(0)
                rho.append((n_act / N).cpu().numpy()); surv.append((n_act > 0).float().cpu().numpy())
                if regions is not None:
                    regt.append(((Rm @ X) / Rsz[:, None]).cpu().numpy())
                    if std is not None:
                        xt.append(((Rx @ xr).mean(1) / Rxs).cpu().numpy())
                        num, den = (Rx @ (X * (ef if spike else xr))).sum(1), (Rx @ X).sum(1)
                        xat.append(torch.where(den > 0, num / den.clamp(min=1), torch.nan).cpu().numpy())
                    if ad is not None:
                        tht.append(((Ra @ th).mean(1) / Ras).cpu().numpy())
                k += 1
                if k == len(rec):
                    break
                if absorbing and n_act.sum() == 0:      # all replicas absorbed: pad and stop (swc.py convention)
                    pad = len(rec) - k
                    rho += [np.zeros(B)] * pad; surv += [np.zeros(B)] * pad
                    if regions is not None:
                        regt += [np.zeros((len(nanR), B))] * pad
                        xt += [nanR] * (pad if std is not None else 0); xat += [nanR] * (pad if std is not None else 0)
                        tht += [nanR] * (pad if ad is not None else 0)
                    break
        if dev == "cuda":
            torch.cuda.synchronize()
        out["sps"].append((time.time() - tg) / max(s_done, 1))
        out["rho"].append(np.array(rho)); out["surv"].append(np.array(surv))
        n_den = max(n_occ, 1) if occ_steps is None else max(occ_steps[1] - occ_steps[0], 1)
        out["occ"].append((oc / n_den).cpu().numpy().astype(np.float16))
        if counts is not None:
            out["ca"].append(c_act.cpu().numpy()); out["co"].append(c_on.cpu().numpy())
            if counts.get("reg"):
                out["cra"].append(c_ract.cpu().numpy()); out["cro"].append(c_ron.cpu().numpy())
        if ramp is not None:
            sd = {"seg_rho": sg_s / rA, "seg_rho2": sg_q / rA, "seg_alive": sg_al}
            if regions is not None:
                sd.update(seg_reg=sg_rs / rA, seg_reg2=sg_rq / rA)
                if std is not None:
                    sd.update(seg_x=sg_x, seg_xa=sg_xa)
            out["seg"].append({k: v.float().cpu().numpy() for k, v in sd.items()})
        if per_step_reg:
            if windows:
                nw = torch.tensor([max(s1 - s0, 1) for s0, s1 in windows], device=dev, dtype=torch.float64)[:, None, None]
                out["wm"].append((wsum / nw).float().cpu().numpy()); out["w2"].append((wsq / nw).float().cpu().numpy())
            if reg_last:
                out["last"].append(last.float().cpu().numpy())
        if regions is not None:
            out["reg"].append(np.array(regt, np.float32))
            if std is not None:
                out["x"].append(np.array(xt, np.float32)); out["xa"].append(np.array(xat, np.float32))
            if ad is not None:
                out["th"].append(np.array(tht, np.float32))
        if log:
            msg = f"g={g:.4f} rho(T)={out['rho'][-1][-1].mean():.4g} surv(T)={out['surv'][-1][-1].mean():.3f}"
            if regions is not None:
                msg += " | " + " ".join(f"{n}={v:.3g}" for n, v in zip(regions["names"], out["reg"][-1][-1].mean(1)))
            log(msg + f" | {1e3 * out['sps'][-1]:.2f} ms/step")
    res = {"rho": np.array(out["rho"]), "surv": np.array(out["surv"]), "occ": np.array(out["occ"]),
           "sec_per_step": np.array(out["sps"]), "t": rec * dt}
    if regions is not None:
        res["reg_rho"] = np.array(out["reg"])                       # [g, t, R, B]
        if std is not None:
            res["res_x"] = np.array(out["x"]); res["res_x_act"] = np.array(out["xa"])
        if ad is not None:
            res["res_theta"] = np.array(out["th"])
    if out["wm"]:
        res["win_mean"] = np.array(out["wm"]); res["win_m2"] = np.array(out["w2"])     # [g, W, R, B]
    if out["last"]:
        res["reg_last"] = np.array(out["last"])                                        # [g, R, B]
    if out["ca"]:
        res["cnt_act"] = np.array(out["ca"]); res["cnt_on"] = np.array(out["co"])       # [g, s, B] int32
        res["cnt_t"] = np.arange(c0 + 1, c1 + 1) * dt
    if out["cra"]:
        res["cnt_reg_act"] = np.array(out["cra"]); res["cnt_reg_on"] = np.array(out["cro"])   # [g, s, R, B] int32
    if out["seg"]:
        res.update(out["seg"][0]); res["seg_g"] = rG.copy()
    return res


# ---------------------------------------------------------------------------------------------------------- main
def group_params(L, N, default, groups, name):
    """Per-neuron (p1, p2) arrays: default (or zeros) overwritten by 'SEL=p1,p2' groups (later wins)."""
    p1 = np.full(N, default[0] if default else 0.0); p2 = np.full(N, default[1] if default else np.inf)
    info = []
    for s in groups:
        sel, v = split_assign(s)
        a, b = (float(x) for x in v.split(","))
        m = select(sel, L); p1[m] = a; p2[m] = b
        info.append({"sel": sel, name[0]: a, name[1]: b, "n": int(m.sum())})
    return p1, p2, info


def main(argv=None):
    a = build_parser().parse_args(argv)
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:.1f}s] {m}", flush=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    side = {"options": vars(a).copy(), "device": dev}
    if dev == "cuda":
        side["gpu"] = torch.cuda.get_device_name(0)

    gsub, gc = None, None
    if a.graph is None:
        pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
        w = np.load(C + "/weight.npy").astype(np.float32); sign = np.load(C + "/sign.npy")
    else:                                                   # Model-2 addition: another graph (e.g. a pd3 FSS graph)
        G_ = np.load(a.graph, allow_pickle=True)
        pre = G_["pre"].astype(np.int64); post = G_["post"].astype(np.int64)
        w = G_["w"].astype(np.float32); sign = G_["sign"]
        gsub = G_["sub"].astype(np.int64) if "sub" in G_.files and G_["sub"].ndim == 1 else None
        gc = float(G_["c"]) if "c" in G_.files else None
        side["graph"] = {"file": a.graph, "N": int(len(sign)), "n_edges": int(len(pre)), "sub": gsub is not None, "c": gc}
    N = len(sign)
    nrng = rng if a.null_seed is None else np.random.default_rng(a.null_seed)
    if a.null and a.null_kind == "targets":
        post = post[nrng.permutation(len(post))]
    elif a.null:
        w = w[nrng.permutation(len(w))]
    if a.null:                                              # identifies the null graph (same hash = same graph)
        import hashlib
        side["null_edges_md5"] = hashlib.md5(np.ascontiguousarray(post).tobytes() + np.ascontiguousarray(w).tobytes()).hexdigest()
    L = Labels(N, sign=sign, sub=gsub)
    if a.graph is not None and gsub is None:
        L.from_cache = False                                # no labels: only all/none/sign:/idx: selectors are meaningful

    # weights and normalisation
    wT = wtransform(w, a.wtransform)
    scale = node_scale(post, w, wT, N, a.norm, a.wnorm)
    if gc is not None and a.norm == "raw" and a.wtransform == "linear":
        scale = np.full(N, gc)                              # the graph file's own raw normalisation (e.g. full-brain c)
    side["c_raw"] = float(scale[0]) if a.norm == "raw" else None
    if a.wtransform != "linear":
        ins_lin = np.bincount(post, weights=w, minlength=N); ins_T = np.bincount(post, weights=wT, minlength=N)
        side["mean_in_strength_model_units"] = float((ins_T * scale)[ins_lin > 0].mean())

    # signs
    overrides = []
    for s in a.sign_override:
        sel, v = split_assign(s)
        m = select(sel, L); overrides.append((m, float(v)))
        side.setdefault("sign_override", []).append({"sel": sel, "V": float(v), "n": int(m.sum()),
                                                     "consensus_signs": {str(k): int(c) for k, c in zip(
                                                         *np.unique(sign[m], return_counts=True))}})
    if a.sign_ensemble > 0:
        tab = nt_table()
        Pr, q = sign_probs(tab, a.ens_conf, a.ens_gt, a.ens_qmiss)
        erng = np.random.default_rng([a.seed if a.ens_seed is None else a.ens_seed, 1])
        types = L.get("type")
        all_draws = draw_signs(Pr, a.sign_ensemble, erng, a.ens_unit, types)
        draw_ids = list(range(a.sign_ensemble)) if a.ens_only is None else list(a.ens_only)
        sign_sets = [(d, all_draws[d].astype(np.float64)) for d in draw_ids]
        mod = L.module()
        side["ensemble"] = {"P_mean": Pr.mean(0).tolist(), "q_mean": float(q.mean()),
                            "draws": [{"draw": d, "n_changed": int((s != np.sign(sign)).sum()),
                                       "n_changed_module": int((s[mod] != np.sign(sign[mod])).sum()),
                                       "lLN1_bc_signs": {str(k): int(c) for k, c in zip(*np.unique(
                                           s[types == "lLN1_bc"], return_counts=True))}} for d, s in sign_sets]}
    else:
        sign_sets = [(None, sign)]
    # Model-2 addition: per-edge sign overrides "PRE->POST=V" (V < 0 scaled by r, as --sign_override)
    edge_ov = []
    for s in a.edge_sign:
        if "->" not in s:
            raise ValueError(f"--edge_sign expects 'PRE->POST=V', got {s!r}")
        pp_, v = split_assign(s)
        ps, qs = (x.strip() for x in pp_.split("->", 1))
        mp, mq = select(ps, L), select(qs, L)
        em = mp[pre] & mq[post]
        if not em.any():
            raise ValueError(f"--edge_sign {s!r} selects no edge")
        v = float(v); ve = v if v >= 0 else v * a.r
        edge_ov.append((em, ve))
        side.setdefault("edge_sign", []).append({
            "pre": ps, "post": qs, "V": v, "sigma_edge": ve, "n_pre": int(mp.sum()), "n_post": int(mq.sum()),
            "n_edges": int(em.sum()), "n_synapses": float(w[em].sum()),
            "pre_consensus_signs_of_edges": {str(k): int(c) for k, c in zip(*np.unique(sign[pre[em]], return_counts=True))}})

    # drive / clamps
    hv = None
    if a.drive:
        hv = np.full(N, a.h, np.float64)
        for s in a.drive:
            sel, v = split_assign(s)
            m = select(sel, L); hv[m] = float(v)
            side.setdefault("drive", []).append({"sel": sel, "h": float(v), "n": int(m.sum())})
    off = np.zeros(N, bool)
    for s in a.clamp_off:
        m = select(s, L); off |= m
        side.setdefault("clamp_off", []).append({"sel": s, "n": int(m.sum())})
    on = np.zeros(N, bool)
    for s in a.clamp_on:
        m = select(s, L); on |= m
        side.setdefault("clamp_on", []).append({"sel": s, "n": int(m.sum())})
    off &= ~on

    # plasticity
    std_spec = adapt_spec = None
    if a.std is not None or a.std_group:
        U, TAU, info = group_params(L, N, a.std, a.std_group, ("U", "tau"))
        if np.any(U < 0) or np.any(TAU <= 0):
            raise ValueError("--std/--std_group: need U >= 0 and TAU > 0 (TAU may be inf)")
        std_spec = (U, TAU); side["std"] = {"default": a.std, "groups": info, "n_depressing": int((U > 0).sum()),
                                            "mode": a.std_mode}
    if a.adapt is not None or a.adapt_group:
        D, TA, info = group_params(L, N, a.adapt, a.adapt_group, ("D", "tau"))
        if np.any((D > 0) & ~(np.isfinite(TA) & (TA > 0))):
            raise ValueError("--adapt/--adapt_group: TAUA must be finite and > 0 where D > 0")
        TA = np.where(D > 0, TA, 1.0)                  # D = 0 units: theta stays 0 whatever TAUA is
        adapt_spec = (D, TA); side["adapt"] = {"default": a.adapt, "groups": info, "n_adapting": int((D > 0).sum())}
    P = make_params(N, a.dt, dev, std_spec, adapt_spec, a.refractory, a.h, hv, off, on, std_mode=a.std_mode)

    # Model-2 addition: quasi-static g schedule in one run (--ramp D [--ramp_loop])
    ramp = None
    if a.ramp > 0:
        sched = list(a.g) + (list(a.g[-2::-1]) if a.ramp_loop else [])
        Ls = int(round(a.ramp / a.dt))
        ramp = {"g": np.array(sched, float), "L": Ls, "A": min(Ls, max(1, int(round(a.ramp_avg * Ls))))}
        a.T = len(sched) * Ls * a.dt
        side["ramp"] = {"schedule": sched, "dwell": a.ramp, "dwell_steps": Ls, "avg_steps": ramp["A"], "T": a.T}

    # E1 options: phase switch, extra records, windows, occupancy window
    steps = int(round(a.T / a.dt))
    tstep = lambda t: int(round(float(t) / a.dt))
    span = lambda s: tuple(tstep(x) for x in s.split(":"))
    switch, extra = None, []
    if a.switch_at is not None:
        s_sw = tstep(a.switch_at)
        if not 1 <= s_sw < steps:
            raise ValueError("--switch_at must lie inside (0, T)")
        hv2 = None
        if hv is not None or a.sw_drive:
            hv2 = hv.copy() if hv is not None else np.full(N, a.h, np.float64)
        for s in a.sw_drive:
            sel, v = split_assign(s)
            m = select(sel, L); hv2[m] = float(v)
            side.setdefault("sw_drive", []).append({"sel": sel, "h": float(v), "n": int(m.sum())})
        off2, on2 = off.copy(), on.copy()
        for s in a.sw_clamp_off:
            m = select(s, L); off2 |= m
            side.setdefault("sw_clamp_off", []).append({"sel": s, "n": int(m.sum())})
        for s in a.sw_clamp_on:
            m = select(s, L); on2 |= m
            side.setdefault("sw_clamp_on", []).append({"sel": s, "n": int(m.sum())})
        off2 &= ~on2
        switch = {"s": s_sw, "P": make_params(N, a.dt, dev, None, None, 0, a.h, hv2, off2, on2)}
        side["switch_step"] = s_sw
        extra += [np.array([s_sw]), s_sw + np.unique(np.round(np.logspace(0, np.log10(max(steps - s_sw, 1)), 150)).astype(int))]
    if a.rec_lin > 0:
        kk = max(1, tstep(a.rec_lin)); extra.append(np.arange(kk, steps + 1, kk))
    if ramp is not None:
        extra.append(ramp["L"] * np.arange(1, len(ramp["g"]) + 1))
    rec_extra = np.concatenate(extra) if extra else None
    windows = [span(s) for s in a.win]
    occ_steps = span(a.occ_win) if a.occ_win else None
    if windows:
        side["windows_steps"] = windows
    counts = None
    if a.rec_counts or a.counts_reg:
        cw = span(a.counts_win) if a.counts_win else (0, steps)
        counts = {"s0": cw[0], "s1": min(cw[1], steps), "reg": bool(a.counts_reg)}
        side["counts"] = {"steps": [counts["s0"], counts["s1"]], "reg": counts["reg"]}
    pulse = None
    if a.pulse is not None:
        pm = select(a.pulse, L) & ~off & ~on
        if not pm.any():
            raise ValueError("--pulse: every selected neuron is clamped")
        gen = torch.Generator(device=dev); gen.manual_seed(int(a.seed) * 7919 + 104729)
        pp = 1.0 if a.pulse_gap <= 0 else float(1 - np.exp(-a.dt / a.pulse_gap))
        pulse = {"idx": torch.tensor(np.where(pm)[0], device=dev), "p": pp, "gen": gen}
        side["pulse"] = {"sel": a.pulse, "n": int(pm.sum()), "gap": a.pulse_gap, "p_per_silent_step": pp,
                         "mean_silent_interval": a.dt / pp}

    init = None
    if a.init:
        if a.exp != "quiet":
            raise ValueError("--init needs --exp quiet")
        im = np.zeros(N, bool)
        for s in a.init:
            im |= select(s, L)
        igen = torch.Generator(device=dev); igen.manual_seed(int(a.seed) * 7919 + 1299709)
        init = {"mask": torch.tensor(im, device=dev)[:, None], "frac": float(a.init_frac), "gen": igen}
        side["init"] = {"sel": a.init, "n": int(im.sum()), "frac": a.init_frac}

    # regions
    no_lab = a.graph is not None and gsub is None
    regs = [] if (a.no_default_regions or no_lab) else [("AL_module", "module"), ("AL_rest", "al_rest"), ("CNS_rest", "cns_rest")]
    regs += [split_assign(s) for s in a.region]
    regions = None
    if regs:
        masks = [select(s, L) for _, s in regs]
        regions = {"names": [n for n, _ in regs], "Rm": torch.tensor(np.array(masks), dtype=torch.float32, device=dev)}
        side["regions"] = {n: {"sel": s, "n": int(m.sum())} for (n, s), m in zip(regs, masks)}
    log(f"setup N={N}, norm={a.norm}, r={a.r}, wtransform={a.wtransform}/{a.wnorm}, null={a.null}, "
        f"draws={len(sign_sets)}, drive={a.drive}, clamp_off={a.clamp_off}, std={side.get('std')}, "
        f"adapt={side.get('adapt')}, refractory={a.refractory}")

    allres, signs_out, nnzs = [], [], []
    for d, sg in sign_sets:
        sig = sigma_vector(sg, a.r, overrides)
        Wm, nnz = build_matrix(pre, post, wT, scale, sig, N, dev, edge_ov=edge_ov or None)
        nnzs.append(nnz)
        log(f"graph on GPU (draw {d}): nnz={nnz}")
        push = PushMV(Wm, N, a.push_max) if a.engine == "push" else None
        res = simulate(Wm, N, a.g if ramp is None else [float(ramp["g"][0])], P, exp=a.exp, T=a.T, dt=a.dt, B=a.B,
                       beta=a.beta, rng=rng, dev=dev,
                       seed_set=a.seed_set, regions=regions, occ=not a.no_occ, log=log, push=push,
                       switch=switch, rec_extra=rec_extra, windows=windows, occ_steps=occ_steps, reg_last=a.reg_last,
                       counts=counts, pulse=pulse, ramp=ramp, init=init)
        if push is not None:
            side.setdefault("engine_steps", []).append(dict(push.stats))
            del push
        allres.append(res)
        if d is not None:
            signs_out.append(sg.astype(np.int8))
        del Wm
    nd = len(allres)
    out = {"t": allres[0]["t"], "g": np.array(a.g),
           "rho": np.concatenate([r["rho"] for r in allres], axis=2),
           "surv": np.concatenate([r["surv"] for r in allres], axis=2),
           "occ": allres[0]["occ"] if nd == 1 else np.mean([r["occ"].astype(np.float32) for r in allres], 0).astype(np.float16),
           "sec_per_step": np.array([r["sec_per_step"] for r in allres])}
    if nd > 1:
        out["occ_ens"] = np.array([r["occ"] for r in allres])
    if regions is not None:
        out["reg_rho"] = np.concatenate([r["reg_rho"] for r in allres], axis=3)
        out["reg_names"] = np.array(regions["names"]); out["reg_sizes"] = regions["Rm"].sum(1).cpu().numpy()
        for k in ("res_x", "res_x_act", "res_theta"):
            if k in allres[0]:
                out[k] = np.array([r[k] for r in allres])
        if "win_mean" in allres[0]:
            out["win_mean"] = np.concatenate([r["win_mean"] for r in allres], axis=3)
            out["win_m2"] = np.concatenate([r["win_m2"] for r in allres], axis=3)
            out["win_bounds"] = np.array([[s0 * a.dt, s1 * a.dt] for s0, s1 in windows])
        if "reg_last" in allres[0]:
            out["reg_last"] = np.concatenate([r["reg_last"] for r in allres], axis=2)
    # Model-2 outputs (replica axis concatenated over sign draws, as rho)
    if "cnt_act" in allres[0]:
        out["cnt_act"] = np.concatenate([r["cnt_act"] for r in allres], axis=2)
        out["cnt_on"] = np.concatenate([r["cnt_on"] for r in allres], axis=2)
        out["cnt_t"] = allres[0]["cnt_t"]
    if "cnt_reg_act" in allres[0]:
        out["cnt_reg_act"] = np.concatenate([r["cnt_reg_act"] for r in allres], axis=3)
        out["cnt_reg_on"] = np.concatenate([r["cnt_reg_on"] for r in allres], axis=3)
    if ramp is not None:
        out["g"] = ramp["g"][:1].copy()
        out["seg_g"] = ramp["g"].copy()
        for k in ("seg_rho", "seg_rho2", "seg_alive"):
            out[k] = np.concatenate([r[k] for r in allres], axis=1)
        for k in ("seg_reg", "seg_reg2"):
            if k in allres[0]:
                out[k] = np.concatenate([r[k] for r in allres], axis=2)
        for k in ("seg_x", "seg_xa"):
            if k in allres[0]:
                out[k] = np.array([r[k] for r in allres]) if nd > 1 else allres[0][k]
    if a.sign_ensemble > 0:
        out["signs_ens"] = np.array(signs_out); out["ens_draw"] = np.repeat([d for d, _ in sign_sets], a.B)
    out["meta"] = json.dumps(vars(a))
    side.update({"N": N, "nnz": nnzs, "steps": int(round(a.T / a.dt)), "n_draws": nd,
                 "ms_per_step": (1e3 * out["sec_per_step"]).tolist(),
                 "rho_T_mean": out["rho"][:, -1, :].mean(1).tolist(), "surv_T": out["surv"][:, -1, :].mean(1).tolist(),
                 "wall_s": time.time() - t0})
    if regions is not None:
        side["reg_rho_T_mean"] = {n: out["reg_rho"][:, -1, i, :].mean(1).tolist() for i, n in enumerate(regions["names"])}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez_compressed(a.out, **out)
    json.dump(side, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1, default=str)
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
