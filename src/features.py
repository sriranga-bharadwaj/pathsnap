"""Features for each (GPS observation, candidate arc) pair.

Only uses the GPS track and the road network (never .route), and only
location-independent quantities (no coordinates, arc ids or trip ids), so the model
can transfer to unseen cities. No window-support features (each row uses only the
observation itself and its immediate neighbours for heading / speed / acceleration).
"""
import numpy as np

from src.geometry import angle_diff

FEATURES = [
    "dist",          # perpendicular distance fix -> arc segment (m)
    "heading_cos",   # cos(movement heading - arc direction): +1 same way, -1 opposite (0 if unknown)
    "heading_valid", # 1 if both movement heading and arc direction are defined
    "frac",          # projection position along the arc, 0 = start node, 1 = end node
    "rank",          # 0 = nearest candidate of this observation, 1 = second nearest, ...
    "rel_dist",      # dist - distance of the nearest candidate of this observation (m)
    "speed",         # m/s from previous fix (clipped to 70)
    "accel",         # m/s^2 (clipped to +-20)
    "dt",            # seconds since previous fix (clipped to 120)
]


def build_features(trk, net, cands):
    """Return X (float32, rows aligned with cands['obs'] / cands['arc'])."""
    obs, arc = cands["obs"], cands["arc"]
    dist = cands["dist"]
    n = len(trk["x"])
    best = np.full(n, np.inf)
    np.minimum.at(best, obs, dist)
    rel_dist = dist - best[obs]
    valid = trk["heading_ok"][obs] & net.heading_valid[arc]
    hcos = np.where(valid, np.cos(angle_diff(trk["heading"][obs], net.heading[arc])), 0.0)
    X = np.column_stack([
        dist,
        hcos,
        valid.astype(float),
        cands["frac"],
        cands["rank"],
        rel_dist,
        np.clip(trk["speed"][obs], 0, 70),
        np.clip(trk["accel"][obs], -20, 20),
        np.clip(trk["dt"][obs], 0, 120),
    ]).astype(np.float32)
    return X
