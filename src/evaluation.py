"""Evaluation metrics. Labels come from src.alignment (offline / server-side only, after prediction).

Metric definitions (used everywhere in this project):

* Candidate-level metrics: every (observation, candidate arc) row is one binary example,
  y = 1 if the candidate is the observation's aligned reference arc. Threshold 0.5 on the
  classifier probability. "Candidate-level macro F1" = mean of the F1 of class 0 and class 1.
  Also accuracy, precision/recall/F1 of class 1, log loss, confusion matrix.
  These are inflated by class imbalance (most rows are negatives), so they are secondary.

* Observation-level Top-1 accuracy (PRIMARY): for each GPS observation pick the candidate
  with the highest score; correct if it equals the aligned reference arc. The denominator
  is ALL observations of the trip(s): observations without candidates ("unmatched") and
  observations whose reference arc is not among the candidates count as errors.

* Candidate miss rate: share of observations whose reference arc is not in their candidate set.

* Route continuity: over consecutive observation pairs inside one segment (no gap > 60 s)
  that both have a predicted arc, the share whose arcs are the same arc, directly
  connected (to_node(A) == from_node(B)), or connected through exactly one intermediate arc.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, confusion_matrix)


def candidate_metrics(y, p, threshold=0.5):
    yhat = (p >= threshold).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(
        cand_accuracy=accuracy_score(y, yhat),
        cand_precision=precision_score(y, yhat, zero_division=0),
        cand_recall=recall_score(y, yhat, zero_division=0),
        cand_f1=f1_score(y, yhat, zero_division=0),
        cand_macro_f1=f1_score(y, yhat, average="macro", zero_division=0),
        log_loss=log_loss(y, pc, labels=[0, 1]),
        confusion=confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    )


def top1_predictions(obs, arc, score, n_obs):
    """Highest-scoring candidate per observation -> (pred_arc, pred_score); -1 / nan if none."""
    pred_arc = np.full(n_obs, -1, dtype=np.int64)
    pred_score = np.full(n_obs, np.nan)
    if len(obs) == 0:
        return pred_arc, pred_score
    order = np.lexsort((-score, obs))             # by obs, then score descending
    o, a, s = obs[order], arc[order], score[order]
    first = np.r_[True, o[1:] != o[:-1]]
    pred_arc[o[first]] = a[first]
    pred_score[o[first]] = s[first]
    return pred_arc, pred_score


def normalise_per_obs(obs, score, n_obs):
    """Scores -> probabilities that sum to 1 over each observation's candidates."""
    tot = np.zeros(n_obs)
    np.add.at(tot, obs, score)
    return score / np.maximum(tot[obs], 1e-12)


def compress(seq):
    """Drop consecutive repeats: [5,5,7,7,7,5] -> [5,7,5]."""
    seq = np.asarray(seq)
    if len(seq) == 0:
        return seq
    return seq[np.r_[True, seq[1:] != seq[:-1]]]


def arc_set_prf(pred_arcs, ref_arcs):
    """Route-level precision / recall / F1 on the SET of arcs (order ignored)."""
    P, R = set(np.asarray(pred_arcs).tolist()), set(np.asarray(ref_arcs).tolist())
    tp = len(P & R)
    prec = tp / len(P) if P else 0.0
    rec = tp / len(R) if R else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def norm_edit_distance(a, b):
    """Levenshtein distance between two arc sequences / max(len). 0 = identical, 1 = disjoint.
    Row-vectorised DP: insertions are resolved with a running minimum of (row - j) + j."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return 1.0
    prev = np.arange(len(b) + 1, dtype=np.int64)
    j = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = np.empty_like(prev)
        cur[0] = i
        sub = prev[:-1] + (b != a[i - 1])          # substitution / match
        dele = prev[1:] + 1                         # deletion
        cur[1:] = np.minimum(sub, dele)
        cur = np.minimum.accumulate(cur - j) + j    # insertion: cur[j] = min(cur[j], cur[j-1] + 1)
        prev = cur
    return float(prev[-1] / max(len(a), len(b)))


def route_continuity(pred_arc, segment, a_from, a_to, n_nodes):
    i = np.arange(len(pred_arc) - 1)
    ok = (segment[i] == segment[i + 1]) & (pred_arc[i] >= 0) & (pred_arc[i + 1] >= 0)
    A, B = pred_arc[i[ok]], pred_arc[i[ok] + 1]
    if len(A) == 0:
        return np.nan
    same = A == B
    direct = a_to[A] == a_from[B]
    # one intermediate arc k with from_k == to_A and to_k == from_B
    arc_codes = np.unique(a_from.astype(np.int64) * n_nodes + a_to)
    two_hop = np.isin(a_to[A].astype(np.int64) * n_nodes + a_from[B], arc_codes)
    return float(np.mean(same | direct | two_hop))
