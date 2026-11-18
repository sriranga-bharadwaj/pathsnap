"""GPS track preprocessing: projection, time steps, speed, movement heading, segments.

Uses only the .track file (allowed in the prediction pipeline).
"""
import numpy as np

from src.data import load_track, make_projection

GAP_BREAK_S = 60.0       # a time gap above this starts a new segment (decoder restarts)
MIN_MOVE_M = 3.0         # displacement below this gives no reliable heading


def prepare_track(trip_dir, trip_id):
    """Load and preprocess one trip's GPS track.

    Returns (track dict, fwd, inv). Track arrays (length N, file order):
      lon, lat, t, x, y      raw and projected positions
      dt                     seconds since previous fix (0 for the first point)
      speed                  m/s from previous fix (0 for the first point of a segment)
      accel                  m/s^2, change of speed per second
      heading, heading_ok    movement direction (radians) from neighbouring fixes, and validity
      segment                segment id; increments after each gap > GAP_BREAK_S
    """
    lon, lat, t = load_track(trip_dir, trip_id)
    fwd, inv = make_projection(lon, lat)
    x, y = map(np.asarray, fwd(lon, lat))
    n = len(x)
    dt = np.r_[0.0, np.diff(t)]
    new_seg = dt > GAP_BREAK_S
    segment = np.cumsum(new_seg)
    step = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, 0.0)
    speed[new_seg] = 0.0
    speed[0] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        accel = np.where(dt > 0, np.r_[0.0, np.diff(speed)] / dt, 0.0)
    accel[new_seg] = 0.0
    accel[0] = 0.0

    # Movement heading: central difference p[i+1] - p[i-1] (one-sided at segment ends).
    prev_i = np.arange(n) - 1
    next_i = np.arange(n) + 1
    prev_i[0] = 0
    next_i[-1] = n - 1
    # do not look across a segment break
    first_of_seg = np.r_[True, segment[1:] != segment[:-1]]
    last_of_seg = np.r_[segment[1:] != segment[:-1], True]
    prev_i[first_of_seg] = np.where(first_of_seg)[0]
    next_i[last_of_seg] = np.where(last_of_seg)[0]
    hx, hy = x[next_i] - x[prev_i], y[next_i] - y[prev_i]
    heading = np.arctan2(hy, hx)
    heading_ok = np.hypot(hx, hy) >= MIN_MOVE_M
    track = dict(lon=lon, lat=lat, t=t, x=x, y=y, dt=dt, speed=speed, accel=accel,
                 heading=heading, heading_ok=heading_ok, segment=segment)
    return track, fwd, inv
