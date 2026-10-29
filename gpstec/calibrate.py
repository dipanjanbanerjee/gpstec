"""Removing the satellite and receiver code biases from leveled slant TEC.

Convention: STEC = STEC_raw + B_sat[prn] + B_rx   (all in TECU)

Satellite biases come from CODE DCB files when available, otherwise they are
estimated from the data together with the receiver bias ("lsq").

Receiver bias:
  minstd  Pick the B_rx that makes the VTEC seen by different satellites at the
          same moment agree best (smallest spread). This is the method used by
          Gopi Seemala's GPS-TEC program (after Ma & Maruyama, 2003).
  lsq     Fit STEC_raw = VTEC(t, ipp)/mf - B_sat - B_rx by least squares, with a
          VTEC that is linear in pierce-point lat/lon in 15-minute blocks. Gives
          satellite and receiver biases with no external files (sum of satellite
          biases = 0, the same convention CODE uses).
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .orbits import gps_seconds


def apply_sat_bias(d, sat_bias):
    d = d.copy()
    d["b_sat"] = d["prn"].map(sat_bias).astype(float)
    missing = sorted(d.loc[d["b_sat"].isna(), "prn"].unique())
    if missing:
        print(f"  warning: no satellite bias for PRN {missing}; those satellites are dropped")
        d = d[d["b_sat"].notna()].copy()
    d["stec_sat"] = d["stec_raw"] + d["b_sat"]
    return d


def _sample(d, el_min, step_s=30):
    s = d[d["el"] >= el_min]
    t = gps_seconds(s["time"].to_numpy())
    return s[np.isclose(t % step_s, 0, atol=0.01)]


def receiver_bias_minstd(d, el_min=30.0, step_s=30, search=(-150.0, 150.0)):
    """Receiver bias (TECU) minimising the spread of simultaneous VTEC values."""
    s = _sample(d, el_min, step_s)
    if s.empty:
        raise ValueError(f"No data above {el_min} deg elevation to estimate the receiver bias")
    epoch = s["time"].to_numpy()
    _, inv = np.unique(epoch, return_inverse=True)
    stec = s["stec_sat"].to_numpy()
    mf = s["mf"].to_numpy()
    nep = inv.max() + 1
    cnt = np.bincount(inv, minlength=nep)
    ok = cnt[inv] >= 2

    def cost(b):
        v = (stec + b) * mf
        m = np.bincount(inv, v, nep) / np.maximum(cnt, 1)
        r = (v - m[inv]) ** 2
        var = np.bincount(inv, np.where(ok, r, 0.0), nep) / np.maximum(cnt, 1)
        return np.sqrt(var[cnt >= 2]).mean()

    grid = np.arange(search[0], search[1] + 1e-9, 1.0)
    c = np.array([cost(b) for b in grid])
    b0 = grid[c.argmin()]
    fine = np.arange(b0 - 1.0, b0 + 1.0 + 1e-9, 0.01)
    cf = np.array([cost(b) for b in fine])
    return float(fine[cf.argmin()]), float(cf.min())


def sample_for_fit(d, el_min=30.0, step_s=30):
    """The small subset of columns/rows the bias fit needs (to combine many days)."""
    return _sample(d, el_min, step_s)[["time", "prn", "el", "mf", "ipp_lat", "ipp_lon", "stec_raw"]].copy()


def biases_lsq(d, el_min=30.0, step_s=30, block_min=15.0, smooth=0.5, presampled=False):
    """Joint least-squares estimate of satellite and receiver biases (TECU).

    Returns (sat_bias dict, receiver_bias, rms of fit).
    """
    s = (d if presampled else _sample(d, el_min, step_s)).reset_index(drop=True)
    if s.empty:
        raise ValueError("No data for the bias fit")
    t = gps_seconds(s["time"].to_numpy())
    blk = ((t - t.min()) // (block_min * 60)).astype(int)
    nb = blk.max() + 1
    prns = np.sort(s["prn"].unique())
    pidx = np.searchsorted(prns, s["prn"].to_numpy())
    ns = len(prns)
    lat0, lon0 = np.median(s["ipp_lat"]), np.median(s["ipp_lon"])
    dlat = s["ipp_lat"].to_numpy() - lat0
    dlon = s["ipp_lon"].to_numpy() - lon0
    inv_mf = 1.0 / s["mf"].to_numpy()
    y = s["stec_raw"].to_numpy()
    n = len(s)
    w = np.sin(np.radians(s["el"].to_numpy()))     # elevation weighting

    # unknown layout: [a0 (nb), a1 (nb), a2 (nb), b_sat (ns), b_rx]
    rows = np.arange(n)
    cols, vals, rr = [], [], []
    for off, coef in ((0, inv_mf), (nb, inv_mf * dlat), (2 * nb, inv_mf * dlon)):
        rr.append(rows); cols.append(off + blk); vals.append(coef)
    rr.append(rows); cols.append(3 * nb + pidx); vals.append(-np.ones(n))
    rr.append(rows); cols.append(np.full(n, 3 * nb + ns)); vals.append(-np.ones(n))
    nun = 3 * nb + ns + 1
    A = sparse.csr_matrix((np.concatenate(vals) * np.tile(w, 5), (np.concatenate(rr), np.concatenate(cols))), shape=(n, nun))
    yw = y * w

    # constraints: sum of satellite biases = 0 (strong), smooth VTEC/gradients in time (weak)
    extra_r, extra_c, extra_v, extra_y = [], [], [], []
    r0 = 0
    big = 1e3
    extra_r += [r0] * ns; extra_c += list(range(3 * nb, 3 * nb + ns)); extra_v += [big] * ns; extra_y.append(0.0); r0 += 1
    for off in (0, nb, 2 * nb):
        for kk in range(nb - 1):
            extra_r += [r0, r0]; extra_c += [off + kk, off + kk + 1]; extra_v += [smooth, -smooth]; extra_y.append(0.0); r0 += 1
    C = sparse.csr_matrix((extra_v, (extra_r, extra_c)), shape=(r0, nun))
    AA = sparse.vstack([A, C]).tocsr()
    yy = np.concatenate([yw, extra_y])
    sol = lsqr(AA, yy, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    resid = A @ sol - yw
    sat = {int(p): float(b) for p, b in zip(prns, sol[3 * nb:3 * nb + ns])}
    return sat, float(sol[-1]), float(np.sqrt(np.mean((resid / w) ** 2)))


def apply_receiver_bias(d, b_rx):
    d = d.copy()
    d["b_rx"] = b_rx
    d["stec"] = d["stec_sat"] + b_rx
    d["vtec"] = d["stec"] * d["mf"]
    return d


def receiver_bias_lsq_fixed_sats(d, sta_lat, sta_lon, el_min=20.0, step_s=30, block_min=15.0, smooth=0.5):
    """Receiver bias (TECU) by least squares when satellite biases are already applied.

    Fits stec_sat = VTEC(t, ipp)/mf - B_rx, with VTEC linear in pierce-point offset
    in 15-minute blocks. Uses the change of the mapping factor with elevation to
    separate B_rx from VTEC; best on night data, where gradients are weak.
    """
    s = _sample(d, el_min, step_s).reset_index(drop=True)
    if len(s) < 50:
        raise ValueError("Not enough data for the least-squares receiver bias")
    t = gps_seconds(s["time"].to_numpy())
    blk = ((t - t.min()) // (block_min * 60)).astype(int)
    nb = blk.max() + 1
    n = len(s)
    r = np.arange(n)
    inv = 1.0 / s["mf"].to_numpy()
    dlat = s["ipp_lat"].to_numpy() - sta_lat
    dlon = (s["ipp_lon"].to_numpy() - sta_lon) * np.cos(np.radians(sta_lat))
    w = np.sin(np.radians(s["el"].to_numpy()))
    rows = np.concatenate([r, r, r, r])
    cols = np.concatenate([blk, nb + blk, 2 * nb + blk, np.full(n, 3 * nb)])
    vals = np.concatenate([inv, inv * dlat, inv * dlon, -np.ones(n)]) * np.tile(w, 4)
    nun = 3 * nb + 1
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, nun))
    er, ec, ev = [], [], []
    k = 0
    for off in (0, nb, 2 * nb):
        for j in range(nb - 1):
            er += [k, k]; ec += [off + j, off + j + 1]; ev += [smooth, -smooth]; k += 1
    C = sparse.csr_matrix((ev, (er, ec)), shape=(k, nun))
    sol = lsqr(sparse.vstack([A, C]).tocsr(), np.r_[s["stec_sat"].to_numpy() * w, np.zeros(k)],
               atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    return float(sol[-1])


def receiver_bias_floor(d, el_min=10.0, pct=99.5):
    """Smallest receiver bias that keeps (almost) every satellite's VTEC >= 0.

    VTEC = (stec_sat + B_rx) * mf >= 0  <=>  B_rx >= -stec_sat. Using the 99.5th
    percentile instead of the maximum ignores the noisiest 0.5 % of points.
    """
    s = d[d["el"] >= el_min]
    return float(np.percentile(-s["stec_sat"].to_numpy(), pct)) if len(s) else -np.inf
