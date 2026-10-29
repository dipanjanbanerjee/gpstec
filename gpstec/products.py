"""Daily VTEC series and output files."""
from pathlib import Path

import numpy as np
import pandas as pd


def clipped_mean(v, nsig=2.0, iters=2):
    """Mean and std after iteratively dropping points beyond nsig standard deviations."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    for _ in range(iters):
        if v.size < 3:
            break
        m, s = v.mean(), v.std()
        keep = np.abs(v - m) <= nsig * s if s > 0 else np.ones(v.size, bool)
        if keep.all():
            break
        v = v[keep]
    return float(v.mean()), float(v.std()) if v.size > 1 else 0.0, int(v.size)


def daily_series(d, day, el_min=50.0, bin_min=1.0, nsig=2.0, min_sats=1):
    """Binned VTEC over one UT day.

    Every bin of the day is present; bins with no satellite above ``el_min`` are
    NaN (gaps are left as gaps, never filled in).
    """
    day = np.datetime64(day, "D")
    s = d[(d["el"] >= el_min) & (d["time"] >= day) & (d["time"] < day + np.timedelta64(1, "D"))]
    hours = (s["time"] - day).dt.total_seconds().to_numpy() / 3600.0
    nb = int(round(24 * 60 / bin_min))
    b = np.floor(hours * 60 / bin_min).astype(int)
    out = pd.DataFrame({"ut_hour": np.arange(nb) * bin_min / 60.0})
    mean = np.full(nb, np.nan); std = np.full(nb, np.nan); nobs = np.zeros(nb, int); nsat = np.zeros(nb, int)
    if len(s):
        # per bin: average each satellite first (1 Hz -> 1 value), then combine satellites
        per_sat = pd.DataFrame({"b": b, "prn": s["prn"].to_numpy(), "v": s["vtec"].to_numpy()}).groupby(["b", "prn"])["v"].mean()
        for bi, grp in per_sat.groupby(level=0):
            if not 0 <= bi < nb:
                continue
            m, sd, n = clipped_mean(grp.to_numpy(), nsig)
            if n >= min_sats:
                mean[bi], std[bi], nsat[bi] = m, sd, n
        cnt = np.bincount(b[(b >= 0) & (b < nb)], minlength=nb)
        nobs[:] = cnt[:nb]
    out["vtec"] = mean
    out["vtec_std"] = std
    out["n_sats"] = nsat
    out["n_obs"] = nobs
    out.insert(0, "date", str(day))
    return out


def write_cmn_like(d, path, station, settings, source=""):
    """Gopi-style .Cmn table (same columns) with the corrected station position."""
    day0 = d["time"].min().normalize()
    t = d["time"]
    mjd = (t - np.datetime64("1858-11-17")).dt.total_seconds() / 86400.0
    hrs = (t - day0).dt.total_seconds() / 3600.0
    tab = pd.DataFrame({
        "MJdatet": mjd.round(6), "Time": hrs.round(6), "PRN": d["prn"],
        "Az": d["az"].round(2), "Ele": d["el"].round(2),
        "Lat": d["ipp_lat"].round(3), "Lon": d["ipp_lon"].round(3),
        "Stec": d["stec"].round(2), "Vtec": d["vtec"].round(2), "S4": -99.0,
    }).sort_values(["PRN", "Time"])
    with open(path, "w", newline="\n") as fh:
        fh.write(f"{station.name},\t\"{source}\"\n")
        fh.write(f"{station.lat:.5f}\t{station.lon:.5f}\t{station.height:.5f}\n")
        tab.to_csv(fh, sep="\t", index=False)


def write_std_like(series, path, station):
    """Gopi-style .Std: UT hour, VTEC (smooth station curve), std between satellites, station latitude."""
    with open(path, "w", newline="\n") as fh:
        for r in series.itertuples():
            v = -99.0 if np.isnan(r.vtec_smooth) else r.vtec_smooth
            s = -99.0 if np.isnan(r.vtec_std) else r.vtec_std
            fh.write(f"{r.ut_hour:.3f} \t{v:.2f} \t{s:.2f} \t{station.lat:.2f}\n")


def write_obs_csv(d, path):
    cols = ["time", "prn", "arc", "az", "el", "ipp_lat", "ipp_lon", "stec", "vtec", "mf"]
    d[cols].to_csv(path, index=False, float_format="%.4f")



def smooth_station_vtec(d, day, sta_lat, sta_lon, el_min=50.0, knot_min=60.0, step_s=30,
                        lam_v=1.0, lam_g=20.0, max_gap_min=30.0, grid_min=1.0):
    """Smooth VTEC above the station, with horizontal gradients removed.

    Every satellite sees the ionosphere at its own pierce point, up to a few hundred
    km from the station. Near the anomaly crest the north-south gradient is strong,
    so satellites disagree by several TECU and a plain average jumps whenever a
    satellite rises or sets. Here all satellites above ``el_min`` are fitted together:

        VTEC_sat(t) = V0(t) + Gn(t) * dlat + Ge(t) * dlon      (dlat, dlon in degrees)

    V0 (VTEC over the station) and the gradients Gn, Ge (TECU/deg) are cubic
    B-splines in time (knots every ``knot_min`` minutes) with a curvature penalty.
    Returned on a ``grid_min`` grid; times more than ``max_gap_min`` from any data are NaN.
    """
    from scipy.interpolate import BSpline
    from scipy import sparse
    from scipy.sparse.linalg import lsqr

    day = np.datetime64(day, "D")
    pad = np.timedelta64(2, "h")              # neighbouring data steadies the day edges
    s = d[(d["el"] >= el_min) & (d["time"] >= day - pad) & (d["time"] < day + np.timedelta64(1, "D") + pad)]
    t = (s["time"] - day).dt.total_seconds().to_numpy()
    s = s[np.isclose(t % step_s, 0, atol=0.01)]
    grid_h = np.arange(0, 24, grid_min / 60.0) + grid_min / 120.0
    out = pd.DataFrame({"ut_hour": grid_h - grid_min / 120.0, "vtec_smooth": np.nan,
                        "grad_north": np.nan, "grad_east": np.nan})
    if len(s) < 20:
        return out
    th = (s["time"] - day).dt.total_seconds().to_numpy() / 3600.0
    lo, hi = th.min(), th.max()
    k = 3
    inner = np.arange(lo, hi + knot_min / 60.0, knot_min / 60.0)
    knots = np.r_[[inner[0]] * k, inner, [inner[-1]] * k]
    nb = len(knots) - k - 1
    B = BSpline.design_matrix(np.clip(th, inner[0], inner[-1]), knots, k).tocsr()
    dlat = s["ipp_lat"].to_numpy() - sta_lat
    dlon = (s["ipp_lon"].to_numpy() - sta_lon) * np.cos(np.radians(sta_lat))
    y = s["vtec"].to_numpy()
    w = np.sin(np.radians(s["el"].to_numpy()))
    A = sparse.diags(w) @ sparse.hstack([B, B.multiply(dlat[:, None]), B.multiply(dlon[:, None])]).tocsr()
    D2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(nb - 2, nb))
    Z2 = sparse.csr_matrix((nb - 2, nb))
    Z = sparse.csr_matrix((nb, nb))
    I = sparse.eye(nb)
    P = sparse.vstack([
        sparse.hstack([lam_v * D2, Z2, Z2]),
        sparse.hstack([Z2, lam_g * D2, Z2]),
        sparse.hstack([Z2, Z2, lam_g * D2]),
        sparse.hstack([Z, 0.3 * I, Z]),          # mild pull of the gradients towards zero
        sparse.hstack([Z, Z, 0.3 * I]),
    ]).tocsr()
    AA = sparse.vstack([A, P]).tocsr()
    yy = np.r_[y * w, np.zeros(P.shape[0])]
    c = lsqr(AA, yy, atol=1e-10, btol=1e-10, iter_lim=50000)[0]
    G = BSpline.design_matrix(np.clip(grid_h, inner[0], inner[-1]), knots, k).tocsr()
    v0, gn, ge = G @ c[:nb], G @ c[nb:2 * nb], G @ c[2 * nb:]
    data_t = np.unique(np.round(th, 3))
    pos = np.clip(np.searchsorted(data_t, grid_h), 1, len(data_t) - 1)
    dist = np.minimum(np.abs(grid_h - data_t[pos - 1]), np.abs(grid_h - data_t[pos]))
    ok = (dist <= max_gap_min / 60.0) & (grid_h >= lo) & (grid_h <= hi)
    out["vtec_smooth"] = np.where(ok, v0, np.nan)
    out["grad_north"] = np.where(ok, gn, np.nan)
    out["grad_east"] = np.where(ok, ge, np.nan)
    return out
