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
