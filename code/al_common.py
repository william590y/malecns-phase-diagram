"""Shared helpers for the antennal-lobe (AL) data-grounding analyses (al_*.py, 2026-11-09).

Labels come from the cache annotation arrays (classes / superclasses / types / nt); the flat MaleCNS release has no
per-synapse neuropil (ROI) table, so "region" here means the region a NEURON is annotated to, and an edge belongs to a
region when both endpoints do. The excitatory operator is the one the simulation uses at r = 0:
M[post, pre] = c * w for presynaptic ACh neurons, c = 1 / (mean in-strength over neurons with input) = 1/742.7.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

C = "/MALECNS_CACHE"
ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/al_lit/grounding"
ALPHA_EFF = (1 - np.exp(-0.2)) / 0.2          # 0.906, synchronous-update death rate (see CLAUDE.md)

AL_CLASSES = ("olfactory", "hygrosensory", "thermosensory", "ALPN", "ALLN", "ALIN", "ALON")


def load(arrays=("pre", "post", "weight", "sign")):
    d = {}
    for a in arrays:
        x = np.load(f"{C}/{a}.npy", allow_pickle=True)
        d[a] = x.astype(np.int64) if a in ("pre", "post") else (x.astype(np.float64) if a == "weight" else x)
    return d


def labels():
    """Per-neuron labels: 'cat' (fine AL category or region), 'reg' (coarse region), 'glom' (glomerulus for ORN/HRN/
    TRN/uPN), and the raw annotation arrays."""
    cl = np.load(f"{C}/classes.npy", allow_pickle=True).astype(str)
    sup = np.load(f"{C}/superclasses.npy", allow_pickle=True).astype(str)
    ty = np.load(f"{C}/types.npy", allow_pickle=True).astype(str)
    nt = np.load(f"{C}/nt.npy", allow_pickle=True).astype(str)
    sides = np.load(f"{C}/sides.npy", allow_pickle=True).astype(str)
    pos = np.load(f"{C}/positions.npy")
    N = len(cl)
    reg = np.full(N, "CB other", object)
    reg[sup == ""] = "unannotated"
    reg[np.isin(sup, ["cb_sensory", "cb_sensory_tbc"])] = "CB sensory other"
    reg[np.isin(cl, ["mechanosensory", "mechanosensory_tactile", "mechanosensory_proprioceptive",
                     "mechanosensory_tbc"]) & np.char.startswith(sup, "cb")] = "JO/mechano (brain)"
    reg[np.isin(cl, ["gustatory", "SEZPN"]) & ~np.char.startswith(sup, "vnc")] = "gustatory/SEZ"
    reg[np.char.startswith(sup, "vnc")] = "VNC"
    reg[np.isin(sup, ["ascending_neuron", "descending_neuron", "sensory_ascending", "sensory_descending",
                      "efferent_ascending", "efferent_descending", "descending_neuron_tbc",
                      "sensory_ascending_tbc"])] = "AN/DN"
    reg[np.isin(sup, ["cb_motor", "cb_endocrine", "cb_efferent", "ENS"])] = "CB motor/endocrine"
    reg[np.isin(sup, ["visual_projection", "visual_centrifugal", "visual_projection_tbc"]) | (cl == "ol_bilateral")] = "VPN/VCN"
    reg[sup == "ol_intrinsic"] = "OL intrinsic"
    reg[sup == "ol_sensory"] = "OL photoreceptors"
    lh = (sup == "cb_intrinsic") & (cl == "") & np.char.startswith(ty, "LH")
    reg[lh] = "LH"
    reg[(cl == "CX")] = "CX"
    reg[np.isin(cl, ["Kenyon_Cell", "MBON", "DAN"]) | np.isin(ty, ["APL", "DPM"])] = "MB"
    reg[np.isin(cl, AL_CLASSES)] = "AL"

    cat = reg.copy()
    cat[cl == "olfactory"] = "AL ORN"
    cat[np.isin(cl, ["hygrosensory", "thermosensory"])] = "AL HRN/TRN"
    pn = cl == "ALPN"
    mpn = pn & (np.char.startswith(ty, "M_") | np.char.startswith(ty, "MZ_") | (np.char.find(ty, "+") >= 0) | (ty == ""))
    cat[pn & ~mpn] = "AL uPN"
    cat[mpn] = "AL mPN"
    ln = cl == "ALLN"
    cat[ln] = "AL LN other"
    cat[ln & np.char.startswith(ty, "lLN1")] = "AL lLN1"
    cat[ln & np.char.startswith(ty, "lLN2")] = "AL lLN2"
    cat[ln & np.char.startswith(ty, "lLN") & ~np.char.startswith(ty, "lLN1") & ~np.char.startswith(ty, "lLN2")] = "AL lLN other"
    cat[ln & (np.char.startswith(ty, "v2LN") | np.char.startswith(ty, "vLN") | np.char.startswith(ty, "il3LN"))] = "AL v/il3LN"
    cat[np.isin(cl, ["ALIN", "ALON"])] = "AL IN/ON"
    cat[(reg == "MB") & (cl == "Kenyon_Cell")] = "MB KC"
    cat[(reg == "MB") & (cl != "Kenyon_Cell")] = "MB MBON/DAN/APL/DPM"

    glom = np.full(N, "", object)
    for i in np.where(np.isin(cl, ["olfactory", "hygrosensory", "thermosensory"]))[0]:
        glom[i] = ty[i].split("_", 1)[1] if "_" in ty[i] else ty[i]
    for i in np.where(pn & ~mpn)[0]:
        glom[i] = ty[i].split("_")[0]
    return dict(cl=cl, sup=sup, ty=ty, nt=nt, sides=sides, pos=pos, reg=reg.astype(str), cat=cat.astype(str),
                glom=glom.astype(str), N=N)


def scale_c(post, w, N):
    ins = np.bincount(post, weights=w, minlength=N)
    return 1.0 / ins[ins > 0].mean()


def exc_operator(d, keep_node=None, keep_edge=None, transform=None, c=None):
    """Directed excitatory operator M[post, pre] (ACh presyn only). c defaults to the simulation's global c from ALL
    edges with the untransformed weights (so lambda stays comparable across variants unless the weights change)."""
    pre, post, w, sign = d["pre"], d["post"], d["weight"], d["sign"]
    N = len(sign)
    if c is None:
        c = scale_c(post, w, N)
    m = sign[pre] > 0
    if keep_node is not None:
        m &= keep_node[pre] & keep_node[post]
    if keep_edge is not None:
        m &= keep_edge
    ww = w[m] if transform is None else transform(w[m])
    return sp.csr_matrix((ww * c, (post[m], pre[m])), shape=(N, N)), c


def pf(M, k=4, left=False, tol=1e-8):
    """Top-k eigenpairs by real part of a nonnegative sparse matrix (right eigenvectors; left=True -> of M^T)."""
    A = M.T.tocsr() if left else M
    vals, vecs = sla.eigs(A, k=k, which="LR", tol=tol, maxiter=200000, ncv=max(2 * k + 1, 24))
    o = np.argsort(-vals.real)
    vals, vecs = vals[o], vecs[:, o]
    P = np.abs(vecs) ** 2
    P /= P.sum(0, keepdims=True)
    return vals, vecs, P


def composition(p, lab, top=None):
    """Shares of probability vector p over the labels in lab (dict label -> share, sorted)."""
    u, inv = np.unique(lab, return_inverse=True)
    s = np.bincount(inv, weights=p, minlength=len(u))
    o = np.argsort(-s)
    out = {str(u[i]): float(s[i]) for i in o if s[i] > 0}
    if top:
        out = dict(list(out.items())[:top])
    return out


def describe_mode(p, L, ntop_types=8):
    """Localization summary of one |v|^2 vector."""
    ipr = float((p ** 2).sum())
    o = np.argsort(-p)
    cs = np.cumsum(p[o])
    sid = L["sides"]
    return {"inv_ipr": 1 / ipr, "n50": int(np.searchsorted(cs, 0.5) + 1), "n90": int(np.searchsorted(cs, 0.9) + 1),
            "n99": int(np.searchsorted(cs, 0.99) + 1),
            "region": composition(p, L["reg"], 6), "cat": composition(p, L["cat"], 8),
            "types": composition(p, L["ty"], ntop_types),
            "side": composition(p, np.where(np.isin(sid, ["L", "R"]), sid, "other")),
            "top_neuron_type": str(L["ty"][o[0]]), "top_neuron_share": float(p[o[0]])}
