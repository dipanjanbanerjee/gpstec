"""Solar flare and geomagnetic storm signatures in the TEC that ``gpstec run`` wrote.

Flares (solar flare effect, SFE)
    A flare's X-rays and EUV ionise the whole sunlit ionosphere within minutes, so
    TEC rises by a fraction of a TECU to a few TECU and then decays over tens of
    minutes. The daily VTEC curve is a fit with knots every hour, so it cannot show
    this. Here the 1-s levelled TEC of every satellite (the ``.Cmn`` files) is used:

    * every continuous satellite pass above the elevation cut (30 deg by default,
      to have several satellites at once) is averaged to 30 s;
    * a straight line is fitted to each pass over the 10 min before the flare start
      and extrapolated; the excess is the VTEC during the 2 min after the X-ray peak
      minus that line. The median over all passes is the flare's TEC rise;
    * the same number is measured at ordinary times (every 5 min within +-1.5 h of
      the flare's UT on every processed day, at least 2 h from any listed flare).
      The p-value is the fraction of ordinary times with an equal or larger rise;
      p < 0.01 counts as detected. Nothing is filtered or removed from the data.

Storms
    For every UT minute the station VTEC is compared with the median of the quiet
    days around it (+-``window_days``). Bounds are median +- 1.5 x interquartile
    range. dVTEC = VTEC - median, also in per cent. Quiet days are those whose
    hourly Dst never goes below ``quiet_dst`` (default -30 nT), or a list you give.
    The CODE global map gets the same treatment at its hourly epochs as a check.
"""
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

MJD0 = np.datetime64("1858-11-17T00:00:00", "ns")
MONTHS = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST",
          "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]


# ─────────────────────────── inputs ───────────────────────────

def find_days(results):
    """{date string: day folder} for every day folder ``gpstec run`` wrote."""
    out = {}
    for p in sorted(Path(results).iterdir()):
        if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name):
            out[p.name] = p
    return out


def station_from_cmn(path):
    """(name, lat, lon, height) from the two header lines of a gpstec .Cmn file."""
    with open(path) as fh:
        name = fh.readline().split(",")[0].strip()
        lat, lon, h = (float(x) for x in fh.readline().split()[:3])
    return name, lat, lon, h


def read_cmn(path, t0=None, t1=None):
    """Per-satellite 1-s TEC from a gpstec ``.Cmn`` file, optionally cut to [t0, t1]."""
    d = pd.read_csv(path, sep="\t", skiprows=2, usecols=["MJdatet", "PRN", "Ele", "Lat", "Lon", "Stec", "Vtec"])
    t = MJD0 + (d["MJdatet"].to_numpy() * 86400.0 * 1e9).astype("int64").astype("timedelta64[ns]")
    d["time"] = pd.to_datetime(t).round("1s")
    d = d.rename(columns={"PRN": "prn", "Ele": "el", "Lat": "ipp_lat", "Lon": "ipp_lon",
                          "Stec": "stec", "Vtec": "vtec"}).drop(columns="MJdatet")
    d = d[(d["stec"] > -90) & (d["vtec"] > -90)]
    if t0 is not None:
        d = d[(d["time"] >= t0) & (d["time"] <= t1)]
    return d.reset_index(drop=True)


def load_window(days, t0, t1):
    """1-s data of every .Cmn file overlapping [t0, t1] (works across midnight)."""
    parts = []
    for day in pd.date_range(pd.Timestamp(t0).normalize(), pd.Timestamp(t1).normalize(), freq="D"):
        folder = days.get(day.strftime("%Y-%m-%d"))
        if folder is None:
            continue
        for f in sorted(folder.glob("*.Cmn")):
            parts.append(read_cmn(f, t0, t1))
    if not parts:
        return pd.DataFrame(columns=["time", "prn", "el", "ipp_lat", "ipp_lon", "stec", "vtec"])
    d = pd.concat(parts, ignore_index=True)
    return d.drop_duplicates(["prn", "time"]).sort_values(["prn", "time"]).reset_index(drop=True)


def read_flares(path):
    """Flare list CSV with columns class, start, peak, end (UTC, ISO 8601). ``end`` may be empty."""
    f = pd.read_csv(path, comment="#", skipinitialspace=True)
    f.columns = [c.strip().lower() for c in f.columns]
    for c in ("start", "peak", "end"):
        if c in f:
            f[c] = pd.to_datetime(f[c], errors="coerce")
    if "end" not in f:
        f["end"] = pd.NaT
    f["end"] = f["end"].fillna(f["peak"] + pd.Timedelta(minutes=20))
    f["start"] = f["start"].fillna(f["peak"] - pd.Timedelta(minutes=10))
    f["class"] = f["class"].astype(str).str.strip()
    return f.dropna(subset=["peak"]).sort_values("peak").reset_index(drop=True)


def parse_dst(text):
    """Hourly Dst (nT) from Kyoto WDC-format lines, a Kyoto monthly page (HTML or text),
    IAGA-2002, or ``time,dst`` CSV. Returns a DataFrame time (centre of the hour), dst."""
    rows = []
    # Kyoto WDC exchange format: DSTyymm*dd ... base, 24 x 4-char hourly values
    for ln in text.splitlines():
        if ln.startswith("DST") and len(ln) >= 116 and ln[7] == "*":
            yy, mm, dd = int(ln[3:5]), int(ln[5:7]), int(ln[8:10])
            cent = ln[14:16]
            year = int(cent) * 100 + yy if cent.strip().isdigit() else 2000 + yy
            base = int(ln[16:20]) if ln[16:20].strip() else 0
            for h in range(24):
                v = int(ln[20 + 4 * h:24 + 4 * h])
                if v != 9999:
                    rows.append((pd.Timestamp(year, mm, dd, h, 30), base * 100 + v))
    if rows:
        return pd.DataFrame(rows, columns=["time", "dst"])
    # Kyoto monthly page: "NOVEMBER   2025" then rows "dd  v1 v2 ... v24"
    plain = re.sub(r"<[^>]+>", " ", text)
    month = year = None
    for ln in plain.splitlines():
        m = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(\d{4})\b", ln.upper())
        if m:
            month, year = MONTHS.index(m.group(1)) + 1, int(m.group(2))
            continue
        m = re.match(r"^\s*(\d{1,2})\s+(-?\d.*)$", ln)
        if month and m:
            vals = [int(v) for v in re.findall(r"-?\d+", m.group(2))]
            if len(vals) == 24:
                day = int(m.group(1))
                rows += [(pd.Timestamp(year, month, day, h, 30), v) for h, v in enumerate(vals) if v != 9999]
    if rows:
        return pd.DataFrame(rows, columns=["time", "dst"])
    # IAGA-2002 or CSV: "YYYY-MM-DD HH:MM[:SS] [DOY] value"
    for ln in text.splitlines():
        m = re.match(r"^\s*(\d{4}-\d{2}-\d{2})[ T,](\d{2}:\d{2})[:\d.]*[\s,]+(?:\d{3}\s+)?(-?\d+(?:\.\d+)?)", ln)
        if m:
            v = float(m.group(3))
            if abs(v) < 9999:
                rows.append((pd.Timestamp(f"{m.group(1)} {m.group(2)}"), v))
    return pd.DataFrame(rows, columns=["time", "dst"])


def download_dst(months, save_dir=None, log=print):
    """Hourly Dst from WDC Kyoto for each 'YYYYMM' (final, then provisional, then real-time)."""
    parts = []
    for ym in months:
        text = None
        for kind in ("dst_final", "dst_provisional", "dst_realtime"):
            url = f"https://wdc.kugi.kyoto-u.ac.jp/{kind}/{ym}/index.html"
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    text = r.read().decode("latin-1")
                if not parse_dst(text).empty:
                    log(f"  Dst {ym}: {url}")
                    break
                text = None
            except Exception:
                text = None
        if text is None:
            log(f"  Dst {ym}: could not download from WDC Kyoto (give a file with --dst FILE)")
            continue
        if save_dir:
            Path(save_dir, f"dst_{ym}.txt").write_text(re.sub(r"<[^>]+>", "", text))
        parts.append(parse_dst(text))
    if not parts:
        return pd.DataFrame(columns=["time", "dst"])
    return pd.concat(parts).drop_duplicates("time").sort_values("time").reset_index(drop=True)


# ─────────────────────────── geometry ───────────────────────────

def solar_zenith(time, lat, lon):
    """Solar zenith angle (deg) and local solar time (h) — NOAA low-precision formulas (~0.1 deg)."""
    t = pd.Timestamp(time)
    doy = t.dayofyear
    hour = t.hour + t.minute / 60 + t.second / 3600
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    lst = (hour * 60 + eqt + 4 * lon) / 60.0 % 24
    ha = np.radians(15 * (lst - 12))
    la = np.radians(lat)
    cz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return float(np.degrees(np.arccos(np.clip(cz, -1, 1)))), float(lst)


# ─────────────────────────── flares ───────────────────────────

def _segments(g, max_gap_s=10, max_step_tecu=0.3):
    """Split one satellite's 1-s data into continuous passes (time gap or a levelling step)."""
    t = g["time"].to_numpy()
    v = g["vtec"].to_numpy()
    dt = np.diff(t).astype("timedelta64[s]").astype(float)
    brk = (dt > max_gap_s) | (np.abs(np.diff(v)) > max_step_tecu * np.maximum(dt, 1))
    seg = np.r_[0, np.cumsum(brk)]
    return [g.iloc[np.flatnonzero(seg == k)] for k in np.unique(seg)]


def passes_30s(d, el_min, step_s=30):
    """30-s means of the 1-s VTEC of every continuous satellite pass above ``el_min``.

    Columns: time (start of the 30-s bin), prn, pass, vtec, el. A bin is kept only
    when it holds at least half of its 1-s samples.
    """
    s = d[d["el"] >= el_min]
    out, k = [], 0
    for prn, g in s.groupby("prn"):
        for seg in _segments(g.sort_values("time")):
            b = seg["time"].dt.floor(f"{step_s}s")
            a = seg.groupby(b).agg(vtec=("vtec", "mean"), el=("el", "mean"), n=("vtec", "size"))
            a = a[a["n"] >= step_s / 2]
            if len(a) < 4:
                continue
            a = a.drop(columns="n").reset_index().rename(columns={"time": "time"})
            a["prn"], a["pass"] = int(prn), k
            k += 1
            out.append(a)
    if not out:
        return pd.DataFrame(columns=["time", "vtec", "el", "prn", "pass"])
    return pd.concat(out, ignore_index=True)


def flare_excess(p30, start, peak, pre_min=10.0, post_min=2.0, min_frac=0.8, step_s=30):
    """VTEC rise during the flare beyond the pre-flare trend, per pass and median over passes.

    For every pass that covers [start - pre_min, peak + post_min]: a straight line is
    fitted to its VTEC over [start - pre_min, start] and extrapolated; the excess is
    the mean VTEC over [peak, peak + post_min] minus the line there.
    Returns (median excess, number of passes, per-pass excesses).
    """
    a0, a1 = start - pd.Timedelta(minutes=pre_min), peak + pd.Timedelta(minutes=post_min)
    w = p30[(p30["time"] >= a0) & (p30["time"] <= a1)]
    need = (a1 - a0).total_seconds() / step_s
    vals = []
    for _, g in w.groupby("pass"):
        if len(g) < min_frac * need or g["time"].min() > a0 + pd.Timedelta(minutes=3) \
                or g["time"].max() < a1 - pd.Timedelta(minutes=2):
            continue
        tm = (g["time"] - start).dt.total_seconds().to_numpy() / 60 + step_s / 120
        v = g["vtec"].to_numpy()
        pre = tm <= 0
        post = tm >= (peak - start).total_seconds() / 60
        if pre.sum() < 5 or post.sum() < 2:
            continue
        c = np.polyfit(tm[pre], v[pre], 1)
        vals.append(float(np.mean(v[post] - np.polyval(c, tm[post]))))
    if len(vals) < 2:
        return np.nan, len(vals), vals
    return float(np.median(vals)), len(vals), vals


def flare_tracks(p30, start, peak, end, half_window_min=90.0, pre_min=10.0, after_end_min=30.0):
    """Per-pass series around a flare for the figure: VTEC, VTEC minus the pre-flare trend
    (from 15 min before the start to 15 min after the X-ray peak) and the rate of TEC (TECU/min)."""
    t_lo = peak - pd.Timedelta(minutes=half_window_min)
    t_hi = peak + pd.Timedelta(minutes=half_window_min)
    w = p30[(p30["time"] >= t_lo) & (p30["time"] <= t_hi)]
    rows = []
    for k, g in w.groupby("pass"):
        g = g.sort_values("time").copy()
        g["min_from_peak"] = (g["time"] - peak).dt.total_seconds() / 60 + 0.25
        tm = (g["time"] - start).dt.total_seconds().to_numpy() / 60 + 0.25
        v = g["vtec"].to_numpy()
        # ROT: change over 60 s (two 30-s bins), placed at the middle
        dtm = np.r_[np.nan, np.nan, tm[2:] - tm[:-2]]
        dv = np.r_[np.nan, np.nan, v[2:] - v[:-2]]
        rot = np.where(np.isclose(dtm, 1.0), dv, np.nan)
        g["rot"] = np.r_[rot[1:], np.nan]
        pre = (tm <= 0) & (tm >= -pre_min)
        g["dvtec"] = np.nan
        if pre.sum() >= 5:
            c = np.polyfit(tm[pre], v[pre], 1)
            # the straight line is only trusted close to where it was fitted
            show = (tm >= -pre_min - 5) & (g["time"] <= peak + pd.Timedelta(minutes=15)).to_numpy()
            g.loc[show, "dvtec"] = v[show] - np.polyval(c, tm[show])
        rows.append(g)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    tr = pd.concat(rows, ignore_index=True)
    q = lambda p: (lambda x: x.quantile(p))
    st = tr.groupby("min_from_peak").agg(
        dvtec=("dvtec", "median"), dvtec_q1=("dvtec", q(0.25)), dvtec_q3=("dvtec", q(0.75)),
        rot=("rot", "median"), rot_q1=("rot", q(0.25)), rot_q3=("rot", q(0.75)),
        n_sats=("prn", "nunique")).reset_index()
    return tr, st


def null_excesses(p30_by_day, start, peak, flare_peaks, span_h=1.5, step_min=5.0, guard_h=2.0, **kw):
    """The same excess measured at ordinary times: every ``step_min`` within +-``span_h`` of
    the flare's UT on every processed day, skipping +-``guard_h`` around any listed flare."""
    rise = peak - start
    ut = peak - peak.normalize()
    vals = []
    for day, p30 in p30_by_day.items():
        base = pd.Timestamp(day) + ut
        for off in np.arange(-span_h * 60, span_h * 60 + 0.1, step_min):
            pk = base + pd.Timedelta(minutes=float(off))
            if any(abs((pk - fp).total_seconds()) < guard_h * 3600 for fp in flare_peaks):
                continue
            v, n, _ = flare_excess(p30, pk - rise, pk, **kw)
            if np.isfinite(v):
                vals.append(v)
    return np.array(vals)


def flare_statistics(excess, n_pass, null, per_pass=()):
    """How unusual the flare-time excess is compared with ordinary times."""
    per_pass = np.asarray(per_pass, float)
    out = {"excess_tecu": round(excess, 3) if np.isfinite(excess) else np.nan, "n_passes": n_pass,
           "fraction_of_passes_rising": round(float(np.mean(per_pass > 0)), 2) if per_pass.size else np.nan,
           "n_null_samples": int(null.size)}
    if null.size < 20 or not np.isfinite(excess):
        out["detected"] = False
        return out
    med = float(np.median(null))
    rstd = float(1.4826 * np.median(np.abs(null - med)))           # robust standard deviation
    out.update({
        "null_median_tecu": round(med, 3),
        "null_robust_std_tecu": round(rstd, 3),
        "null_p99_tecu": round(float(np.quantile(null, 0.99)), 3),
        "z_score": round((excess - med) / rstd, 2) if rstd > 0 else np.nan,
        "p_value": round(float((np.sum(null >= excess) + 1) / (null.size + 1)), 4),
    })
    out["detected"] = bool(out["p_value"] < 0.01)
    return out

# ─────────────────────────── storms ───────────────────────────

def storm_deviation(allv, quiet_days, window_days=7, min_quiet=5, col="vtec_smooth", gim_col="vtec_gim"):
    """dVTEC of each day against the median of nearby quiet days, minute by minute."""
    allv = allv.copy()
    allv["date"] = pd.to_datetime(allv["date"]).dt.strftime("%Y-%m-%d")
    allv["minute"] = np.round(allv["ut_hour"] * 60).astype(int)
    if col not in allv or allv[col].isna().all():
        col = "vtec"
    piv = allv.pivot_table(index="date", columns="minute", values=col, aggfunc="first")
    has_gim = gim_col in allv and allv[gim_col].notna().any()
    gpiv = allv.pivot_table(index="date", columns="minute", values=gim_col, aggfunc="first") if has_gim else None
    dates = pd.to_datetime(piv.index)
    quiet = pd.to_datetime(sorted(set(quiet_days) & set(piv.index)))
    rows = []
    for k, day in enumerate(dates):
        w = window_days
        near = quiet[np.abs((quiet - day).days) <= w]
        while len(near) < min_quiet and w < 60 and len(near) < len(quiet):
            w += 1
            near = quiet[np.abs((quiet - day).days) <= w]
        near = near[near != day] if len(near) > min_quiet else near
        ref = piv.loc[near.strftime("%Y-%m-%d")]
        med = ref.median(axis=0)
        iqr = ref.quantile(0.75, axis=0) - ref.quantile(0.25, axis=0)
        r = pd.DataFrame({"date": day.strftime("%Y-%m-%d"), "ut_hour": piv.columns / 60.0,
                          "vtec": piv.iloc[k].to_numpy(), "quiet_median": med.to_numpy(),
                          "lower_bound": (med - 1.5 * iqr).to_numpy(), "upper_bound": (med + 1.5 * iqr).to_numpy(),
                          "n_quiet_days": len(near)})
        if has_gim:
            gref = gpiv.loc[near.strftime("%Y-%m-%d")]
            r["vtec_gim"] = gpiv.iloc[k].reindex(piv.columns).to_numpy()
            r["gim_quiet_median"] = gref.median(axis=0).reindex(piv.columns).to_numpy()
        rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    out["dvtec"] = out["vtec"] - out["quiet_median"]
    out["dvtec_percent"] = 100 * out["dvtec"] / out["quiet_median"].where(out["quiet_median"] > 2)
    out["outside_bounds"] = (out["vtec"] > out["upper_bound"]) | (out["vtec"] < out["lower_bound"])
    if has_gim:
        out["gim_dvtec"] = out["vtec_gim"] - out["gim_quiet_median"]
        out["gim_dvtec_percent"] = 100 * out["gim_dvtec"] / out["gim_quiet_median"].where(out["gim_quiet_median"] > 2)
    out["time"] = pd.to_datetime(out["date"]) + pd.to_timedelta(out["ut_hour"], unit="h")
    return out


def quiet_days_from_dst(dst, dates, threshold=-30.0):
    """Days whose hourly Dst stays at or above ``threshold`` nT all day."""
    if dst is None or dst.empty:
        return []
    m = dst.groupby(dst["time"].dt.strftime("%Y-%m-%d"))["dst"].agg(["min", "count"])
    return [d for d in dates if d in m.index and m.loc[d, "count"] >= 20 and m.loc[d, "min"] >= threshold]


def daily_storm_table(dev, dst=None):
    """One line per day: Dst minimum, VTEC maximum vs quiet, largest +/- deviation, minutes outside bounds."""
    rows = []
    for day, g in dev.groupby("date"):
        ok = g["vtec"].notna()
        r = {"date": day, "max_vtec": g["vtec"].max(), "quiet_median_at_that_time": np.nan,
             "max_positive_dvtec": g["dvtec"].max(), "max_negative_dvtec": g["dvtec"].min(),
             "max_positive_dvtec_percent": g["dvtec_percent"].max(),
             "max_negative_dvtec_percent": g["dvtec_percent"].min(),
             "minutes_above_upper_bound": int((g["vtec"] > g["upper_bound"]).sum()),
             "minutes_below_lower_bound": int((g["vtec"] < g["lower_bound"]).sum()),
             "n_quiet_days": int(g["n_quiet_days"].iloc[0])}
        if ok.any():
            r["quiet_median_at_that_time"] = g.loc[g["vtec"].idxmax(), "quiet_median"]
        if dst is not None and not dst.empty:
            dd = dst[dst["time"].dt.strftime("%Y-%m-%d") == day]["dst"]
            r["dst_min_nt"] = dd.min() if len(dd) else np.nan
        rows.append(r)
    return pd.DataFrame(rows).round(2)


# ─────────────────────────── driver ───────────────────────────

def run(results, flares_csv=None, dst=None, quiet_days=None, quiet_dst=-30.0, window_days=7,
        el_min=30.0, station=None, out_dir=None, settings=None, log=print):
    from . import plots
    results = Path(results)
    out = Path(out_dir) if out_dir else results / "events"
    out.mkdir(parents=True, exist_ok=True)
    days = find_days(results)
    if not days:
        raise SystemExit(f"No day folders (YYYY-MM-DD) found in {results}. Give the folder that `gpstec run -o` wrote.")
    cmn = next((f for p in days.values() for f in sorted(p.glob("*.Cmn"))), None)
    if station is not None:
        name, lat, lon = station.name, station.lat, station.lon
    elif cmn is not None:
        name, lat, lon, _ = station_from_cmn(cmn)
    else:
        raise SystemExit("No .Cmn files found; give the station with -c station.toml.")
    log(f"gpstec events: {name} ({lat:.3f}N {lon:.3f}E), {len(days)} days in {results}")

    # Dst
    dst_df = None
    if dst:
        if str(dst).lower() == "kyoto":
            months = sorted({d[:4] + d[5:7] for d in days})
            dst_df = download_dst(months, save_dir=out, log=log)
        else:
            dst_df = parse_dst(Path(dst).read_text(errors="replace"))
            log(f"  Dst: {len(dst_df)} hourly values from {dst}")
        if dst_df is not None and dst_df.empty:
            dst_df = None
        if dst_df is not None:
            dst_df.to_csv(out / "dst_hourly.csv", index=False)

    flares = read_flares(flares_csv) if flares_csv else None

    # ── storm ──
    allv_path = results / f"{name}_vtec_all_days.csv"
    if allv_path.exists():
        allv = pd.read_csv(allv_path)
        dates = sorted(pd.to_datetime(allv["date"]).dt.strftime("%Y-%m-%d").unique())
        if quiet_days:
            q = [d for d in quiet_days if d in dates]
            why = "given on the command line"
        elif dst_df is not None:
            q = quiet_days_from_dst(dst_df, dates, quiet_dst)
            why = f"Dst >= {quiet_dst:g} nT all day"
        else:
            q = list(dates)
            why = "all days (no Dst given)"
        if len(q) < 3:
            log(f"  only {len(q)} quiet days ({why}); using all days for the reference")
            q, why = list(dates), "all days"
        log(f"  quiet reference: {len(q)} days ({why}): {', '.join(q)}")
        dev = storm_deviation(allv, q, window_days=window_days)
        dev.drop(columns="time").to_csv(out / f"{name}_storm_dvtec.csv", index=False, float_format="%.3f")
        tab = daily_storm_table(dev, dst_df)
        tab.to_csv(out / f"{name}_storm_daily.csv", index=False)
        log(tab.to_string(index=False))
        try:
            plots.plot_storm(dev, dst_df, flares, q, name, settings, out / f"{name}_storm_dvtec.png")
            if dst_df is not None:
                inside = dst_df[(dst_df["time"] >= dev["time"].min()) & (dst_df["time"] <= dev["time"].max())]
                if len(inside) and inside["dst"].min() <= -50:
                    tmin = inside.loc[inside["dst"].idxmin(), "time"]
                    lo, hi = tmin.normalize() - pd.Timedelta(days=3), tmin.normalize() + pd.Timedelta(days=4)
                    plots.plot_storm(dev, dst_df, flares, q, name, settings,
                                     out / f"{name}_storm_dvtec_{tmin:%Y-%m-%d}.png", t0=lo, t1=hi)
        except Exception as e:
            log(f"  storm plot skipped ({type(e).__name__}: {e})")
    else:
        log(f"  {allv_path.name} not found: storm analysis skipped")

    # ── flares ──
    if flares is not None:
        cache = out / "_cache"
        cache.mkdir(exist_ok=True)
        p30_by_day = {}
        for k, (day, folder) in enumerate(days.items()):
            cf = cache / f"{day}_passes30s_el{el_min:g}.pkl"
            if cf.exists():
                p = pd.read_pickle(cf)
            else:
                f = next(iter(sorted(folder.glob("*.Cmn"))), None)
                if f is None:
                    continue
                log(f"  reading {f.name}")
                p = passes_30s(read_cmn(f), el_min)
                p.to_pickle(cf)
            p = p.copy()
            p["pass"] = p["pass"] + 100000 * k
            p30_by_day[day] = p
        p30_all = pd.concat(p30_by_day.values(), ignore_index=True) if p30_by_day else pd.DataFrame()
        peaks = list(flares["peak"])
        summ, stacks, nulls = [], [], []
        for f in flares.itertuples():
            sza, lst = solar_zenith(f.peak, lat, lon)
            row = {"class": flares.loc[f.Index, "class"], "start": f.start, "peak": f.peak, "end": f.end,
                   "solar_zenith_deg": round(sza, 1), "local_solar_time_h": round(lst, 2)}
            if sza > 95:
                row["note"] = "station in darkness: no effect expected"
                summ.append(row)
                log(f"  {row['class']:>6} {f.peak:%Y-%m-%d %H:%M} UT  night side, skipped")
                continue
            if p30_all.empty or f.peak.strftime("%Y-%m-%d") not in p30_by_day:
                row["note"] = "day not processed"
                summ.append(row)
                log(f"  {row['class']:>6} {f.peak:%Y-%m-%d %H:%M} UT  no data for this day")
                continue
            exc, npass, per = flare_excess(p30_all, f.start, f.peak)
            null = null_excesses(p30_by_day, f.start, f.peak, peaks)
            row.update(flare_statistics(exc, npass, null, per))
            row["excess_per_pass_tecu"] = " ".join(f"{v:+.2f}" for v in per)
            tracks, st = flare_tracks(p30_all, f.start, f.peak, f.end)
            if not np.isfinite(exc):
                row["note"] = f"fewer than 2 satellites above {el_min:g} deg cover the flare"
            else:
                row["note"] = "detected (p < 0.01)" if row["detected"] else "not distinguishable from normal variability"
            tag = f"{f.peak:%Y-%m-%d_%H%M}_{row['class']}"
            if not st.empty:
                tracks.to_csv(out / f"{name}_flare_{tag}_satellites.csv", index=False, float_format="%.4f")
                st.to_csv(out / f"{name}_flare_{tag}_median.csv", index=False, float_format="%.4f")
                try:
                    plots.plot_flare(tracks, st, row, name, el_min, settings, out / f"{name}_flare_{tag}.png")
                except Exception as e:
                    log(f"  flare plot skipped ({type(e).__name__}: {e})")
                if np.isfinite(exc):
                    stacks.append((row, st))
                    nulls.append(null)
            summ.append(row)
            log(f"  {row['class']:>6} {f.peak:%Y-%m-%d %H:%M} UT  SZA {sza:4.1f}  excess {exc:+.2f} TECU "
                f"({npass} sats)  normal times {row.get('null_median_tecu', np.nan):+.2f} "
                f"± {row.get('null_robust_std_tecu', np.nan):.2f}  z {row.get('z_score', np.nan):+.1f}  "
                f"p {row.get('p_value', np.nan):.3f}  -> {row['note']}")
        pd.DataFrame(summ).to_csv(out / f"{name}_flare_summary.csv", index=False)
        if stacks:
            try:
                plots.plot_flare_overview(stacks, nulls, name, el_min, settings, out / f"{name}_flares_overview.png")
            except Exception as e:
                log(f"  flare overview plot skipped ({type(e).__name__}: {e})")
    log(f"  results in {out}")
    return out
