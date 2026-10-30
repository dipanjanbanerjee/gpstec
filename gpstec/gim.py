"""Global ionosphere maps (IONEX), used as an independent, absolute reference.

CODE publishes a daily map CODG{DDD}0.{YY}I (rapid: CORG). At the station it gives
VTEC every 1-2 hours with an accuracy of a few TECU. gpstec uses it to

  * draw a reference line on the daily plot, and
  * report how far the station VTEC is from it at night (calibration check), and
  * optionally (receiver_bias = "gim") set the receiver bias so that the
    station's night-time VTEC matches the map.
"""
from pathlib import Path
import gzip
import urllib.request

import numpy as np
import pandas as pd


def candidate_urls(day):
    day = np.datetime64(day, "D").astype(object)
    y, yy, doy = day.year, day.year % 100, day.timetuple().tm_yday
    short = f"CODG{doy:03d}0.{yy:02d}I.Z"
    long_ = f"COD0OPSFIN_{y}{doy:03d}0000_01D_01H_GIM.INX.gz"
    out = []
    for proto in ("http://www.aiub.unibe.ch/download", "http://ftp.aiub.unibe.ch", "ftp://ftp.aiub.unibe.ch"):
        out.append((short, f"{proto}/CODE/{y}/{short}"))
        out.append((long_, f"{proto}/CODE/{y}/{long_}"))
    return out


def local_files_for(day, folder):
    day = np.datetime64(day, "D").astype(object)
    y, yy, doy = day.year, day.year % 100, day.timetuple().tm_yday
    pats = [f"CODG{doy:03d}0.{yy:02d}I", f"CORG{doy:03d}0.{yy:02d}I", f"COD0OPSFIN_{y}{doy:03d}",
            f"COD0OPSRAP_{y}{doy:03d}", f"IGSG{doy:03d}0.{yy:02d}I"]
    out = []
    for p in sorted(Path(folder).glob("*")):
        n = p.name.upper()
        if any(n.startswith(x.upper()) for x in pats):
            out.append(p)
    return out


def find_or_fetch(day, folder, log=print, timeout=20, offline=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    have = local_files_for(day, folder)
    if have or offline:
        return have[0] if have else None
    for name, url in candidate_urls(day):
        target = folder / name
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                target.write_bytes(r.read())
            log(f"  downloaded {url}")
            return target
        except Exception as e:
            log(f"  not reachable: {url} ({type(e).__name__})")
    return None


def _text(path):
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x9d":
        import unlzw3
        raw = unlzw3.unlzw(raw)
    elif raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("ascii", errors="replace").splitlines()


def read_ionex(path):
    """Return (epochs datetime64[ns] (n,), lats (nlat,), lons (nlon,), tec (n, nlat, nlon) TECU)."""
    lines = _text(path)
    exponent = -1
    i = 0
    while i < len(lines):
        ln = lines[i]
        lab = ln[60:].strip()
        if lab == "EXPONENT":
            exponent = int(ln[:6])
        elif lab == "END OF HEADER":
            i += 1
            break
        i += 1
    epochs, maps = [], []
    lats, lons = None, None
    cur, cur_rows, cur_lats = None, [], []
    while i < len(lines):
        ln = lines[i]
        lab = ln[60:].strip()
        if lab == "START OF TEC MAP":
            cur_rows, cur_lats = [], []
        elif lab == "EPOCH OF CURRENT MAP":
            y, mo, d, h, mi, s = (int(v) for v in ln[:36].split())
            cur = np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}", "ns")
        elif lab == "LAT/LON1/LON2/DLON/H":
            lat = float(ln[2:8]); lon1 = float(ln[8:14]); lon2 = float(ln[14:20]); dlon = float(ln[20:26])
            n = int(round((lon2 - lon1) / dlon)) + 1
            vals = []
            i += 1
            while len(vals) < n:
                row = lines[i]
                vals += [int(row[k:k + 5]) for k in range(0, len(row.rstrip()), 5)]
                i += 1
            cur_rows.append(vals[:n])
            cur_lats.append(lat)
            if lons is None:
                lons = lon1 + dlon * np.arange(n)
            continue
        elif lab == "END OF TEC MAP":
            epochs.append(cur)
            maps.append(np.array(cur_rows, float))
            lats = np.array(cur_lats)
        elif lab == "START OF RMS MAP":
            break
        i += 1
    tec = np.array(maps) * 10.0 ** exponent
    tec[tec >= 9999 * 10.0 ** exponent] = np.nan
    return np.array(epochs, dtype="datetime64[ns]"), lats, lons, tec


def station_maps(path, lat, lon, start, end, method="bilinear"):
    """VTEC at the station from each map whose epoch lies in [start, end] - no time interpolation.

    CODE publishes one map per hour on a 2.5 deg x 5 deg grid. Each returned value is one
    real map. In space it is either the bilinear value from the 4 grid nodes around the
    station ("bilinear", the IONEX standard) or the value of the nearest node ("nearest").
    Returns (epochs as datetime64[ns], values, info dict).
    """
    ep, lats, lons, tec = read_ionex(path)
    start, end = np.datetime64(start, "ns"), np.datetime64(end, "ns")
    sel = np.where((ep >= start) & (ep <= end))[0]
    if method == "nearest":
        i = int(np.argmin(np.abs(lats - lat)))
        j = int(np.argmin(np.abs(((lons - lon + 180) % 360) - 180)))
        vals = tec[sel, i, j]
        node = (float(lats[i]), float(lons[j]))
    else:
        vals = station_series(path, lat, lon, ep[sel])
        node = None
    step = float((ep[1] - ep[0]) / np.timedelta64(1, "s")) if len(ep) > 1 else np.nan
    info = {"interval_s": step, "dlat": float(abs(lats[1] - lats[0])), "dlon": float(abs(lons[1] - lons[0])),
            "method": method, "nearest_node": node}
    return ep[sel], np.asarray(vals, float), info


def station_series(path, lat, lon, times):
    """GIM VTEC at (lat, lon) for the given datetime64 times.

    Bilinear in space; in time the two nearest maps are combined after rotating
    each by the Earth's rotation (the standard IONEX interpolation).
    """
    ep, lats, lons, tec = read_ionex(path)
    th = (ep - ep[0]).astype("timedelta64[s]").astype(float)
    tq = (np.asarray(times, dtype="datetime64[ns]") - ep[0]).astype("timedelta64[s]").astype(float)

    def at(k, lon_q):
        la = np.interp(lat, lats[::-1], np.arange(len(lats))[::-1]) if lats[0] > lats[-1] else np.interp(lat, lats, np.arange(len(lats)))
        lo = ((lon_q - lons[0]) % 360) / (lons[1] - lons[0])
        i0, j0 = int(np.floor(la)), np.floor(lo).astype(int)
        fa, fo = la - i0, lo - j0
        i1 = min(i0 + 1, len(lats) - 1)
        j0 %= len(lons); j1 = (j0 + 1) % len(lons)
        g = tec[k]
        return ((1 - fa) * ((1 - fo) * g[i0, j0] + fo * g[i0, j1]) + fa * ((1 - fo) * g[i1, j0] + fo * g[i1, j1]))

    out = np.full(len(tq), np.nan)
    for n, t in enumerate(tq):
        k = np.searchsorted(th, t) - 1
        if k < 0 or k >= len(th) - 1:
            if 0 <= t <= th[-1] + 1:
                k = min(max(k, 0), len(th) - 2)
            else:
                continue
        t0, t1 = th[k], th[k + 1]
        lon0 = lon + 360.0 * (t - t0) / 86400.0
        lon1 = lon + 360.0 * (t - t1) / 86400.0
        w = (t - t0) / (t1 - t0)
        out[n] = (1 - w) * at(k, lon0) + w * at(k + 1, lon1)
    return out
