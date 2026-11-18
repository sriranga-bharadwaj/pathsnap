"""Candidate generation: which road arcs could each GPS observation be on?

Rule (prediction pipeline, never reads .route):
  1. all arcs within RADIUS metres of the fix (STRtree 'dwithin' query);
  2. keep the K_MAX nearest; if a kept arc's reverse twin was also within the radius,
     keep the twin too (both directions of a road are always offered together, the
     classifier/heading decides the direction);
  3. if no arc is within RADIUS, ONE retry with RETRY_RADIUS;
  4. still empty -> observation is "unmatched" (counted as an error in evaluation).
"""
import numpy as np

from src.geometry import point_to_segments

RADIUS = 30.0         # chosen from the 15/30/50 m sweep in experiments/01 (see results/phase1)
K_MAX = 24
RETRY_RADIUS = 100.0


def generate_candidates(trk, net, radius=RADIUS, k_max=K_MAX, retry_radius=RETRY_RADIUS):
    """Return a dict of flat arrays, one row per (observation, candidate arc) pair:
       obs, arc, dist, frac, rank (0 = nearest), retry (bool: found only by the wider retry)
    plus 'unmatched': boolean array (N,) for observations with no candidate at all."""
    x, y = trk["x"], trk["y"]
    n = len(x)
    pi, ai = net.arcs_within(x, y, radius)
    retry_flag = np.zeros(len(pi), dtype=bool)
    empty = np.setdiff1d(np.arange(n), pi)
    if len(empty) and retry_radius and retry_radius > radius:
        pj, aj = net.arcs_within(x[empty], y[empty], retry_radius)
        pi = np.r_[pi, empty[pj]]
        ai = np.r_[ai, aj]
        retry_flag = np.r_[retry_flag, np.ones(len(pj), dtype=bool)]
    d, f = point_to_segments(x[pi], y[pi], net.x0[ai], net.y0[ai], net.x1[ai], net.y1[ai])

    # sort by observation, then distance, then arc id (deterministic)
    order = np.lexsort((ai, d, pi))
    pi, ai, d, f, retry_flag = pi[order], ai[order], d[order], f[order], retry_flag[order]
    # rank within each observation
    starts = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]
    counts = np.diff(np.r_[starts, len(pi)])
    rank = np.arange(len(pi)) - np.repeat(starts, counts)

    keep = rank < k_max
    if (~keep).any():
        # re-admit reverse twins of kept arcs that fell just outside the cap
        # encode (obs, arc) pairs as one integer so the test is a vectorised isin
        kept_codes = pi[keep].astype(np.int64) * net.n_arcs + ai[keep]
        tw = net.twin[ai]
        cand = ~keep & (tw >= 0)
        twin_codes = pi[cand].astype(np.int64) * net.n_arcs + tw[cand]
        keep[np.flatnonzero(cand)[np.isin(twin_codes, kept_codes)]] = True
    pi, ai, d, f, rank, retry_flag = pi[keep], ai[keep], d[keep], f[keep], rank[keep], retry_flag[keep]
    unmatched = np.ones(n, dtype=bool)
    unmatched[pi] = False
    return dict(obs=pi, arc=ai, dist=d, frac=f, rank=rank, retry=retry_flag, unmatched=unmatched)
