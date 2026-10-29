"""Slant TEC from dual-frequency code and phase, with phase-to-code levelling.

Model, per satellite arc (all in metres, geometry-free combinations):

    P4 = P2 - P1         = I + B_sat + B_rx + code noise/multipath
    L4 = L1*l1 - L2*l2   = I + N (constant per unbroken arc) + small noise

The phase gives a precise but ambiguous TEC; the code gives an absolute but noisy
one. "Levelling" shifts each phase arc onto the code by the (elevation-weighted)
mean of P4 - L4, which removes N. What is left are the satellite and receiver
differential code biases (DCBs), handled in calibrate.py.

    STEC_raw (TECU) = K * L4_levelled,   K = f1^2 f2^2 / (40.3 (f1^2 - f2^2)) / 1e16
"""
import numpy as np
import pandas as pd

from . import constants as k
from .geometry import az_el, mapping_factor, pierce_point
from .orbits import gps_seconds

PAIRS = {
    # name: (code1 candidates, code2, phase1, phase2, f1, f2, K)
    "L1L2": (("P1", "C1"), "P2", "L1", "L2", k.F1, k.F2, k.K_L1L2),
    "L1L5": (("C1", "P1"), "C5", "L1", "L5", k.F1, k.F5, k.K_L1L5),
}


def add_geometry(df, orbit, station, shell_km):
    """Add sat ECEF, azimuth, elevation, IPP lat/lon and the mapping factor."""
    sta_xyz = station.ecef
    times = df["time"].to_numpy()
    prns = df["prn"].to_numpy()
    # Signal travel time is ~0.07 s; the receiver clock in this data is off by a
    # large amount, so use a nominal travel time instead of the pseudorange.
    tau = 0.075
    t_tx = times - np.timedelta64(int(tau * 1e9), "ns")
    # Positions are smooth: evaluate every 30 s and interpolate to 1 Hz for speed.
    sat = _positions_interp(orbit, t_tx, prns, step_s=30.0)
    # Earth rotation during signal travel (Sagnac)
    th = k.OMEGA_E * tau
    rot = np.array([[np.cos(th), np.sin(th), 0], [-np.sin(th), np.cos(th), 0], [0, 0, 1]])
    sat = sat @ rot.T
    az, el = az_el(sta_xyz, station.lat, station.lon, sat)
    df = df.copy()
    df["az"] = az
    df["el"] = el
    ilat, ilon = pierce_point(station.lat, station.lon, az, el, shell_km)
    df["ipp_lat"] = ilat
    df["ipp_lon"] = ilon
    df["mf"] = mapping_factor(el, shell_km)          # VTEC = STEC * mf
    return df


def _positions_interp(orbit, times, prns, step_s=30.0):
    ts = gps_seconds(times)
    out = np.full((len(ts), 3), np.nan)
    for prn in np.unique(prns):
        sel = np.where(prns == prn)[0]
        t = ts[sel]
        nodes = np.arange(np.floor(t.min() / step_s) * step_s - 4 * step_s,
                          t.max() + 5 * step_s, step_s)
        node_times = (np.datetime64("1980-01-06T00:00:00", "ns")
                      + (nodes * 1e9).astype(np.int64).astype("timedelta64[ns]"))
        xyz = orbit.positions(node_times, np.full(len(nodes), prn))
        ok = np.isfinite(xyz[:, 0])
        if ok.sum() < 4:
            continue
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(nodes[ok], xyz[ok], axis=0, extrapolate=False)
        out[sel] = cs(t)
    return out


def compute_stec(df, pair="L1L2", *, max_gap_s=60.0, slip_m=0.25, min_arc_s=600.0,
                 level_el_deg=20.0, mask_el_deg=10.0):
    """Leveled slant TEC (without DCBs) for one signal pair.

    Adds columns: code (which code was used for the first frequency), arc,
    p4, l4, stec_raw, and drops rows that cannot be used.
    """
    code1_opts, code2, ph1, ph2, f1, f2, K = PAIRS[pair]
    code1 = next((c for c in code1_opts if c in df and df[c].notna().any()), None)
    need = [code1, code2, ph1, ph2]
    if code1 is None or any(c not in df for c in need):
        raise ValueError(f"Observations for {pair} are missing (have: {list(df.columns)})")
    lam1, lam2 = k.C / f1, k.C / f2

    d = df.dropna(subset=need).copy()
    d = d[d["el"] >= mask_el_deg]
    d = d.sort_values(["prn", "time"]).reset_index(drop=True)
    d["code"] = code1
    d["p4"] = d[code2] - d[code1]
    d["l4"] = d[ph1] * lam1 - d[ph2] * lam2

    # ── arc detection: data gaps, loss-of-lock flags, jumps in L4 ──
    t = gps_seconds(d["time"].to_numpy())
    prn = d["prn"].to_numpy()
    l4 = d["l4"].to_numpy()
    dt = np.diff(t, prepend=np.nan)
    dl4 = np.diff(l4, prepend=np.nan)
    new_prn = np.r_[True, prn[1:] != prn[:-1]]
    lli = np.zeros(len(d), bool)
    for c in ("LLI" + ph1[1], "LLI" + ph2[1]):
        if c in d:
            lli |= (d[c].to_numpy() & 1) > 0
    # predicted L4 change from the local median rate -> catches cycle slips
    rate = pd.Series(np.where(new_prn, np.nan, dl4 / np.where(dt > 0, dt, np.nan)))
    med_rate = rate.groupby(prn).transform(lambda s: s.rolling(31, center=True, min_periods=5).median()).to_numpy()
    jump = np.abs(dl4 - med_rate * dt) > slip_m * np.maximum(dt, 1.0) ** 0.5
    brk = new_prn | (dt > max_gap_s) | lli | np.nan_to_num(jump, nan=False).astype(bool)
    d["arc"] = np.cumsum(brk)

    # ── drop short arcs ──
    g = d.groupby("arc")["time"]
    span = (g.transform("max") - g.transform("min")).dt.total_seconds()
    d = d[span >= min_arc_s].copy()

    # ── levelling: weighted mean of P4 - L4 over the higher-elevation part of the arc ──
    diff = d["p4"] - d["l4"]
    w = np.sin(np.radians(d["el"])) ** 2
    use = d["el"] >= level_el_deg
    d["_w"] = np.where(use, w, 0.0)
    d["_wd"] = d["_w"] * diff
    # robust: reject points more than 4 sigma from the arc median before averaging
    med = diff.groupby(d["arc"]).transform("median")
    mad = (diff - med).abs().groupby(d["arc"]).transform("median") * 1.4826 + 1e-3
    keep = ((diff - med).abs() < 4 * mad).astype(float)
    d["_w"] *= keep
    d["_wd"] *= keep
    sw = d.groupby("arc")["_w"].transform("sum")
    swd = d.groupby("arc")["_wd"].transform("sum")
    # arcs that never rise above level_el_deg: fall back to a plain median
    offset = np.where(sw > 0, swd / sw.replace(0, np.nan), med)
    d["level_offset"] = offset
    d["l4_lev"] = d["l4"] + offset
    d["stec_raw"] = K * d["l4_lev"]
    d["stec_code"] = K * d["p4"]
    # arc quality: scatter of code around the levelled phase (TECU)
    d["arc_rms"] = (K * (d["p4"] - d["l4_lev"])).groupby(d["arc"]).transform(lambda s: np.sqrt(np.mean(s ** 2)))
    return d.drop(columns=["_w", "_wd"])


def _robust_offset(p4, l4, el, level_el_deg):
    """The levelling offset of one arc (same rule as compute_stec)."""
    diff = p4 - l4
    med = np.median(diff)
    mad = np.median(np.abs(diff - med)) * 1.4826 + 1e-3
    w = np.where(el >= level_el_deg, np.sin(np.radians(el)) ** 2, 0.0) * (np.abs(diff - med) < 4 * mad)
    return float(np.sum(w * diff) / np.sum(w)) if np.sum(w) > 0 else float(med)


def join_midnight_arcs(a, b, pair="L1L2", max_gap_s=60.0, slip_m=0.25, level_el_deg=20.0):
    """Level satellite passes that run from the end of file ``a`` into file ``b`` as one pass.

    The receiver keeps tracking through midnight and the carrier phase continues, so
    the two pieces are one arc. Levelled separately, each piece only sees part of the
    pass and the two offsets can differ by a TECU or more. Both frames are changed in
    place. Returns a list of (prn, old offset a, old offset b, joint offset) in TECU.
    """
    K = PAIRS[pair][6]
    a_end, b_start = a["time"].max(), b["time"].min()
    gap = (b_start - a_end).total_seconds()
    if not (0 < gap <= max_gap_s):
        return []
    done = []
    for prn in sorted(set(a["prn"]) & set(b["prn"])):
        ia = a.index[a["prn"] == prn]
        ib = b.index[b["prn"] == prn]
        arc_a = a.at[ia[-1], "arc"]
        arc_b = b.at[ib[0], "arc"]
        ra = a.loc[ia][a.loc[ia, "arc"] == arc_a]
        rb = b.loc[ib][b.loc[ib, "arc"] == arc_b]
        dt = (rb["time"].iloc[0] - ra["time"].iloc[-1]).total_seconds()
        if dt <= 0 or dt > max_gap_s:
            continue
        if ra["time"].iloc[-1] < a_end - pd.Timedelta(seconds=max_gap_s):
            continue
        # phase must continue: compare the step with the local rate on both sides
        rate = (ra["l4"].iloc[-1] - ra["l4"].iloc[max(-31, -len(ra))]) / max(
            (ra["time"].iloc[-1] - ra["time"].iloc[max(-31, -len(ra))]).total_seconds(), 1.0)
        step = rb["l4"].iloc[0] - ra["l4"].iloc[-1] - rate * dt
        if abs(step) > slip_m * max(dt, 1.0) ** 0.5:
            continue
        both = pd.concat([ra, rb])
        off = _robust_offset(both["p4"].to_numpy(), both["l4"].to_numpy(), both["el"].to_numpy(), level_el_deg)
        old_a, old_b = float(ra["level_offset"].iloc[0]), float(rb["level_offset"].iloc[0])
        for df, idx in ((a, ra.index), (b, rb.index)):
            df.loc[idx, "level_offset"] = off
            df.loc[idx, "l4_lev"] = df.loc[idx, "l4"] + off
            df.loc[idx, "stec_raw"] = K * df.loc[idx, "l4_lev"]
        rms = float(np.sqrt(np.mean((K * (both["p4"] - both["l4"] - off)) ** 2)))
        a.loc[ra.index, "arc_rms"] = rms
        b.loc[rb.index, "arc_rms"] = rms
        done.append((int(prn), K * old_a, K * old_b, K * off))
    return done
