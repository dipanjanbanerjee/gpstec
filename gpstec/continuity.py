"""Midnight continuity check between consecutive days.

The receiver file of day X ends at 00:00:00 of day X+1 and the next file starts one
second later, so the two files see the same satellites through the same patch of
ionosphere at the same moment. The real VTEC cannot jump in one second, so any
difference at midnight comes from the processing:

  * the receiver bias, which each day estimates on its own, and
  * the phase levelling of arcs that cross midnight, which is done separately in
    each file with only part of the arc.

For every satellite seen on both sides, each file's VTEC is fitted with a straight
line over its last (or first) 10 minutes and evaluated at the boundary, and the
two values are compared. The median over satellites is the "jump". The part that
the receiver-bias difference explains is reported separately.
"""
import numpy as np
import pandas as pd

EDGE_MIN = 10.0
MAX_GAP_S = 300.0


def edges(d, el_min=30.0, edge_min=EDGE_MIN):
    """Rows near the start and end of one file, kept for the boundary comparison."""
    t0, t1 = d["time"].min(), d["time"].max()
    w = pd.Timedelta(minutes=edge_min)
    m = (d["el"] >= el_min) & d["vtec"].notna() & ((d["time"] <= t0 + w) | (d["time"] >= t1 - w))
    cols = ["time", "prn", "el", "vtec", "stec", "arc"]
    e = d.loc[m, [c for c in cols if c in d]].copy()
    return {"start": t0, "end": t1, "rows": e}


def _value_at(rows, t_ref):
    """Linear fit of VTEC against time, evaluated at t_ref; None if too few points."""
    if len(rows) < 20:
        return None
    x = (rows["time"] - t_ref).dt.total_seconds().to_numpy()
    y = rows["vtec"].to_numpy()
    if np.ptp(x) < 60:
        return float(np.median(y))
    a, b = np.polyfit(x, y, 1)
    return float(b)


def compare(prev, nxt, el_out, mf_out=0.9):
    """Compare two consecutive files. ``prev``/``nxt`` are dicts with keys
    edges (from :func:`edges`), b_rx, date_end_series, series (daily product DataFrames)."""
    ea, eb = prev["edges"], nxt["edges"]
    gap = (eb["start"] - ea["end"]).total_seconds()
    if not (-3600 <= gap <= MAX_GAP_S):
        return None
    t_ref = ea["end"] + (eb["start"] - ea["end"]) / 2
    w = pd.Timedelta(minutes=EDGE_MIN)
    ra = ea["rows"][ea["rows"]["time"] >= ea["end"] - w]
    rb = eb["rows"][eb["rows"]["time"] <= eb["start"] + w]
    per = []
    for prn in sorted(set(ra["prn"]) & set(rb["prn"])):
        a, b = ra[ra["prn"] == prn], rb[rb["prn"] == prn]
        va, vb = _value_at(a, t_ref), _value_at(b, t_ref)
        if va is None or vb is None:
            continue
        el = float(np.median(np.r_[a["el"].to_numpy()[-5:], b["el"].to_numpy()[:5]]))
        mf = float(np.median(np.r_[(a["vtec"] / a["stec"]).to_numpy()[-5:], (b["vtec"] / b["stec"]).to_numpy()[:5]]))
        per.append({"prn": int(prn), "el": round(el, 1), "vtec_end_prev": round(va, 2),
                    "vtec_start_next": round(vb, 2), "jump": round(vb - va, 2),
                    "jump_from_rx_bias": round((nxt["b_rx"] - prev["b_rx"]) * mf, 2)})
    per = pd.DataFrame(per)
    res = {"boundary_ut": str(t_ref.floor("s")), "prev_date": prev["date"], "next_date": nxt["date"],
           "file_gap_s": round(gap, 1), "n_sats_el30": int(len(per)),
           "receiver_bias_prev": round(prev["b_rx"], 2), "receiver_bias_next": round(nxt["b_rx"], 2)}
    if len(per):
        hi = per[per["el"] >= el_out]
        res["jump_median_el30"] = round(float(per["jump"].median()), 2)
        res["jump_spread_el30"] = round(float(per["jump"].std(ddof=0)), 2) if len(per) > 1 else 0.0
        res["n_sats_out"] = int(len(hi))
        res["jump_median_out"] = round(float(hi["jump"].median()), 2) if len(hi) else None
        res["jump_from_rx_bias"] = round(float(per["jump_from_rx_bias"].median()), 2)
        res["jump_from_levelling"] = round(res["jump_median_el30"] - res["jump_from_rx_bias"], 2)
    # smooth station curve: last value of day X vs first value of day X+1
    sa, sb = prev["series"], nxt["series"]
    if sa is not None and sb is not None:
        va = sa.dropna(subset=["vtec_smooth"])
        vb = sb.dropna(subset=["vtec_smooth"])
        if len(va) and len(vb) and va["ut_hour"].iloc[-1] > 23.5 and vb["ut_hour"].iloc[0] < 0.5:
            res["smooth_end_prev"] = round(float(va["vtec_smooth"].iloc[-1]), 2)
            res["smooth_start_next"] = round(float(vb["vtec_smooth"].iloc[0]), 2)
            res["smooth_jump"] = round(res["smooth_start_next"] - res["smooth_end_prev"], 2)
    unc = [u for u in (prev.get("unc"), nxt.get("unc")) if u]
    tol = max(1.0, float(np.hypot(*unc)) if len(unc) == 2 else (unc[0] if unc else 1.0))
    res["tolerance_tecu"] = round(tol, 2)
    j = res.get("jump_median_el30")
    if j is None:
        res["status"] = "no common satellites"
    else:
        res["status"] = "OK" if abs(j) <= tol else "CHECK"
    return res, per
