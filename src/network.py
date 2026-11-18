"""Per-trip directed road network in projected metres, with a spatial index.

Each trip has its OWN network (.nodes/.arcs), so everything here is built per trip.
Arc geometry is a straight segment from node[from] to node[to] (the files contain
no intermediate shape points).
"""
import numpy as np
import shapely
from scipy.sparse import csr_matrix

from src.data import load_nodes, load_arcs


class TripNetwork:
    def __init__(self, trip_dir, trip_id, fwd):
        """fwd: projection function from src.data.make_projection (built from the GPS track)."""
        nlon, nlat = load_nodes(trip_dir, trip_id)
        self.a_from, self.a_to = load_arcs(trip_dir, trip_id)
        self.n_nodes, self.n_arcs = len(nlon), len(self.a_from)
        self.node_lon, self.node_lat = nlon, nlat
        nx_, ny_ = fwd(nlon, nlat)
        self.nx, self.ny = np.asarray(nx_), np.asarray(ny_)
        # segment end points per arc
        self.x0, self.y0 = self.nx[self.a_from], self.ny[self.a_from]
        self.x1, self.y1 = self.nx[self.a_to], self.ny[self.a_to]
        self.length = np.hypot(self.x1 - self.x0, self.y1 - self.y0)
        # direction of travel along the arc (math convention: radians from +x axis)
        self.heading = np.arctan2(self.y1 - self.y0, self.x1 - self.x0)
        self.heading_valid = self.length > 0.5          # zero-length arcs have no direction
        # reverse twin: arc (to -> from) if it exists, else -1
        key = self.a_from * self.n_nodes + self.a_to
        rkey = self.a_to * self.n_nodes + self.a_from
        order = np.argsort(key)
        pos = np.searchsorted(key[order], rkey)
        pos = np.clip(pos, 0, self.n_arcs - 1)
        found = key[order][pos] == rkey
        self.twin = np.where(found, order[pos], -1)
        # spatial index (STRtree over arc segments); built once per trip
        seg = np.stack([np.c_[self.x0, self.y0], np.c_[self.x1, self.y1]], axis=1)
        self.tree = shapely.STRtree(shapely.linestrings(seg))
        self._graph = None

    def graph(self):
        """Sparse directed node graph weighted by arc length (built lazily, used by the HMM)."""
        if self._graph is None:
            w = np.maximum(self.length, 1e-3)              # csgraph treats 0 as "no edge"
            self._graph = csr_matrix((w, (self.a_from, self.a_to)), shape=(self.n_nodes, self.n_nodes))
        return self._graph

    def arcs_within(self, x, y, radius):
        """All (point_index, arc_id) pairs with arc within `radius` metres of point (x[i], y[i])."""
        pi, ai = self.tree.query(shapely.points(x, y), predicate="dwithin", distance=radius)
        return pi, ai
