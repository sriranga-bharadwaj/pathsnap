"""HMM / Viterbi route reconstruction (prediction pipeline: never reads .route).

Hidden state  : the candidate arc of an observation (each candidate keeps its arc id and
                its projection fraction on that arc, so state <-> road segment is preserved).
Emission      : log P(obs | arc) = alpha * log(p), p = classifier probability normalised over
                the observation's candidates.
Transition    : log P(b | a) = -| route_distance(a -> b) - straight_distance(fix_i, fix_j) | / beta
                (Newson & Krumm style: on a real route the distance driven along the network
                is close to the distance between the two fixes).
  route_distance:
    same arc, moving forward (or back <= BACKWARD_TOL m: GPS noise)  -> |frac_b - frac_a| * len
    reverse twin (U-turn on the same road)                            -> |(1-frac_b) - frac_a| * len,
                                                                         plus REVERSAL penalty
    otherwise: rest of arc a + shortest path(to_node(a) -> from_node(b)) + frac_b * len(b),
               bounded Dijkstra with cutoff = min(V_MAX * dt + TOL, HARD_CAP) (cached per node)
  types: direct (same arc or b starts where a ends), multi_hop (>= 1 intermediate arc),
         reversal, relaxed (see fallbacks)
Segments      : a time gap > 60 s (src/preprocess.py) starts a new segment -> decoder restarts.
Fallbacks     : observation with no candidates -> "unmatched" (skipped; the chain links the
                fixes around it). No valid transition at all into a step ->
                level 1: retry that step with RELAX_CUTOFF; level 2: "jump" (restart from the
                best previous state with a fixed penalty). Both are tagged "relaxed". Never raises.
"""
import heapq

import numpy as np

V_MAX = 50.0          # m/s (180 km/h) - physical bound used for the Dijkstra cutoff
TOL = 150.0           # m - GPS noise + junction slack added to the cutoff
HARD_CAP = 3000.0     # m - normal cutoff never exceeds this
RELAX_CUTOFF = 10000.0  # m - level-1 relaxation cutoff
BACKWARD_TOL = 15.0   # m - tolerated backward movement on the same arc (GPS noise)
JUMP_PENALTY = 50.0   # log-units - level-2 relaxation (restart) penalty
DEFAULT_PARAMS = dict(alpha=1.0, beta=10.0, reversal_penalty=5.0)

T_DIRECT, T_MULTI, T_REVERSAL = 0, 1, 2
TYPE_NAMES = {T_DIRECT: "direct", T_MULTI: "multi_hop", T_REVERSAL: "reversal"}


class RouteGraph:
    """Adjacency lists + cached bounded Dijkstra for one trip network."""

    def __init__(self, net):
        self.net = net
        order = np.argsort(net.a_from, kind="stable")
        self.ptr = np.searchsorted(net.a_from[order], np.arange(net.n_nodes + 1)).tolist()
        self.out_arcs = order.tolist()
        self.to = net.a_to.tolist()
        self.length = net.length.tolist()
        self.cache = {}    # source node -> (limit, dist dict, pred-arc dict)

    def sp(self, s, limit):
        c = self.cache.get(s)
        if c is not None and c[0] >= limit:
            return c[1], c[2]
        dist, pred = {s: 0.0}, {}
        heap = [(0.0, s)]
        done = set()
        ptr, out, to, ln = self.ptr, self.out_arcs, self.to, self.length
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for k in range(ptr[u], ptr[u + 1]):
                a = out[k]
                v = to[a]
                nd = d + ln[a]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    pred[v] = a
                    heapq.heappush(heap, (nd, v))
        self.cache[s] = (limit, dist, pred)
        return dist, pred

    def path_arcs(self, s, t, limit):
        """Arc ids on the shortest path s -> t (empty if s == t). Reuses the cached search
        from s when it already reached t (it did, during decoding); else searches to `limit`."""
        c = self.cache.get(s)
        pred = c[2] if (c is not None and t in c[1]) else self.sp(s, limit)[1]
        arcs = []
        node = t
        while node != s:
            a = pred[node]
            arcs.append(a)
            node = self.net.a_from[a]
        return arcs[::-1]


def step_table(g, A, FA, B, FB, cutoff):
    """Route distances (len(A) x len(B), inf = unreachable) and transition types."""
    net = g.net
    D = np.full((len(A), len(B)), np.inf)
    T = np.full((len(A), len(B)), T_DIRECT, dtype=np.int8)
    for i, (a, fa) in enumerate(zip(A, FA)):
        La = net.length[a]
        rem = (1.0 - fa) * La
        s = int(net.a_to[a])
        dist = None
        for j, (b, fb) in enumerate(zip(B, FB)):
            if b == a and (fa - fb) * La <= BACKWARD_TOL:
                D[i, j] = abs(fb - fa) * La
                continue
            if b == net.twin[a]:
                D[i, j] = abs((1.0 - fb) - fa) * La
                T[i, j] = T_REVERSAL
                continue
            if rem > cutoff:
                continue
            if dist is None:
                dist, _ = g.sp(s, cutoff - rem)
            sb = int(net.a_from[b])
            d = dist.get(sb)
            if d is not None and rem + d <= cutoff:
                D[i, j] = rem + d + fb * net.length[b]
                T[i, j] = T_DIRECT if sb == s else T_MULTI
    return D, T


def precompute(trk, net, cands, g=None):
    """Everything the decoder needs that does NOT depend on alpha / beta / penalties.

    Returns list of segments; each segment: dict(obs, rows, steps) where rows[k] are the
    candidate-row indices of matched observation obs[k], and steps[k-1] = (D, T, gc, relax1).
    """
    g = g or RouteGraph(net)
    obs_all = cands["obs"]
    n = len(trk["x"])
    starts = np.searchsorted(obs_all, np.arange(n))
    ends = np.searchsorted(obs_all, np.arange(n), side="right")
    segs = []
    for sid in np.unique(trk["segment"]):
        o = np.flatnonzero((trk["segment"] == sid) & (ends > starts))   # matched obs only
        if len(o) == 0:
            continue
        rows = [np.arange(starts[i], ends[i]) for i in o]
        steps = []
        for k in range(1, len(o)):
            i, j = o[k - 1], o[k]
            gc = float(np.hypot(trk["x"][j] - trk["x"][i], trk["y"][j] - trk["y"][i]))
            dt = float(trk["t"][j] - trk["t"][i])
            cutoff = min(V_MAX * dt + TOL, HARD_CAP)
            A, FA = cands["arc"][rows[k - 1]], cands["frac"][rows[k - 1]]
            B, FB = cands["arc"][rows[k]], cands["frac"][rows[k]]
            D, T = step_table(g, A, FA, B, FB, cutoff)
            relax1 = False
            if not np.isfinite(D).any():
                D, T = step_table(g, A, FA, B, FB, RELAX_CUTOFF)
                relax1 = True
            steps.append((D, T, gc, relax1))
        segs.append(dict(obs=o, rows=rows, steps=steps))
    return segs, g


def viterbi(segs, prob_norm, params):
    """Decode. Returns per matched obs: chosen candidate row, transition type label."""
    alpha, beta, rp = params["alpha"], params["beta"], params["reversal_penalty"]
    logp = alpha * np.log(np.maximum(prob_norm, 1e-6))
    chosen_row, ttype = {}, {}
    for seg in segs:
        rows, steps = seg["rows"], seg["steps"]
        delta = logp[rows[0]]
        back, relaxed = [], []
        for k, (D, T, gc, relax1) in enumerate(steps):
            logT = -np.abs(D - gc) / beta - rp * (T == T_REVERSAL)   # inf in D -> -inf
            score = delta[:, None] + logT
            best_prev = np.argmax(score, axis=0)
            best = score[best_prev, np.arange(score.shape[1])]
            jump = not np.isfinite(best).any()
            if jump:          # level-2 relaxation: restart from best previous state
                best_prev = np.full(score.shape[1], int(np.argmax(delta)))
                best = np.full(score.shape[1], delta.max() - JUMP_PENALTY)
            back.append(best_prev)
            relaxed.append(jump or relax1)
            delta = best + logp[rows[k + 1]]
        # backtrack
        idx = int(np.argmax(delta))
        picks = [idx]
        for k in range(len(steps) - 1, -1, -1):
            idx = int(back[k][idx])
            picks.append(idx)
        picks = picks[::-1]
        for k, o in enumerate(seg["obs"]):
            chosen_row[o] = rows[k][picks[k]]
            if k == 0:
                ttype[o] = "start"
            elif relaxed[k - 1]:
                ttype[o] = "relaxed"
            else:
                D, T, _, _ = steps[k - 1]
                ttype[o] = TYPE_NAMES[int(T[picks[k - 1], picks[k]])]
    return chosen_row, ttype


def decode(trk, net, cands, prob, params=None, segs=None, g=None):
    """Full decode for one trip. prob = raw classifier probability per candidate row.

    Returns dict:
      arc[N] (-1 unmatched), confidence[N], margin[N], transition_type[N], fallback[N],
      path_arcs: list of arc-id lists (one per continuous piece of the predicted route),
      events: counts.
    """
    params = params or DEFAULT_PARAMS
    n = len(trk["x"])
    obs = cands["obs"]
    tot = np.zeros(n)
    np.add.at(tot, obs, prob)
    pn = prob / np.maximum(tot[obs], 1e-12)
    if segs is None:
        segs, g = precompute(trk, net, cands, g)
    chosen_row, ttype = viterbi(segs, pn, params)

    arc = np.full(n, -1, dtype=np.int64)
    conf = np.zeros(n)
    margin = np.zeros(n)
    tt = np.array(["unmatched"] * n, dtype=object)
    best_other = np.zeros(n)
    for o, r in chosen_row.items():
        arc[o] = cands["arc"][r]
        conf[o] = pn[r]
        tt[o] = ttype[o]
    # margin = chosen normalised prob - best other candidate's normalised prob
    for o, r in chosen_row.items():
        s, e = np.searchsorted(obs, o), np.searchsorted(obs, o, side="right")
        others = np.r_[pn[s:r], pn[r + 1:e]]
        best_other[o] = others.max() if len(others) else 0.0
    margin = np.where(arc >= 0, conf - best_other, 0.0)
    retry = np.zeros(n, dtype=bool)
    if "retry" in cands:
        retry[obs[cands["retry"]]] = True
    fallback = (tt == "relaxed") | (tt == "unmatched") | retry

    # predicted route as arc sequences (fill multi-hop gaps with the shortest-path arcs)
    path_arcs = []
    for seg in segs:
        piece = []
        prev = None
        for o in seg["obs"]:
            r = chosen_row[o]
            a, fa = int(cands["arc"][r]), float(cands["frac"][r])
            if prev is None or ttype[o] == "relaxed":
                if piece:
                    path_arcs.append(piece)
                piece = [a]
            elif a != prev[0]:
                if ttype[o] == "multi_hop":
                    s = int(net.a_to[prev[0]])
                    try:
                        piece += g.path_arcs(s, int(net.a_from[a]), RELAX_CUTOFF)
                    except KeyError:      # should not happen; never crash
                        pass
                piece.append(a)
            prev = (a, fa)
        if piece:
            path_arcs.append(piece)
    events = dict(unmatched_points=int((tt == "unmatched").sum()),
                  relaxed_transitions=int((tt == "relaxed").sum()),
                  multi_hop_transitions=int((tt == "multi_hop").sum()),
                  direct_transitions=int((tt == "direct").sum()),
                  reversal_transitions=int((tt == "reversal").sum()),
                  segment_starts=int((tt == "start").sum()))
    return dict(arc=arc, confidence=conf, margin=margin, transition_type=tt, fallback=fallback,
                path_arcs=path_arcs, events=events)
