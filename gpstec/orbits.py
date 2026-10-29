"""Satellite positions from RINEX navigation files (broadcast ephemeris) or SP3 files.

Every orbit source exposes ``positions(times, prns) -> (N, 3) ECEF metres`` where
``times`` are numpy datetime64 GPS times of signal transmission.
"""
import gzip
import numpy as np
import pandas as pd

from .constants import GM, OMEGA_E

GPS_EPOCH = np.datetime64("1980-01-06T00:00:00", "ns")


def gps_seconds(times):
    """Seconds since the GPS epoch (float) for datetime64 values."""
    return (np.asarray(times, dtype="datetime64[ns]") - GPS_EPOCH).astype(np.int64) / 1e9


def _open(path):
    path = str(path)
    return gzip.open(path, "rt", errors="replace") if path.endswith(".gz") else open(path, "r", errors="replace")


def _d(s):
    s = s.strip().replace("D", "E").replace("d", "E")
    return float(s) if s else 0.0


# ───────────────────────── broadcast ephemeris ─────────────────────────
_EPH_FIELDS = [
    "af0", "af1", "af2",
    "iode", "crs", "dn", "m0",
    "cuc", "e", "cus", "sqrta",
    "toe", "cic", "omega0", "cis",
    "i0", "crc", "omega", "omegadot",
    "idot", "codes", "week", "l2p",
    "sva", "svh", "tgd", "iodc",
    "tot", "fit",
]


def read_nav(path):
    """Read a RINEX 2 or 3 GPS navigation file into a DataFrame (one row per ephemeris)."""
    rows = []
    with _open(path) as fh:
        version = 2.0
        while True:
            line = fh.readline()
            if not line:
                break
            if "RINEX VERSION / TYPE" in line:
                version = float(line[:9])
            if "END OF HEADER" in line:
                break
        lines = fh.readlines()
    i = 0
    if version < 3:
        width, first_off, val_off = 22, 22, 3
    else:
        width, first_off, val_off = 23, 23, 4
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if version >= 3:
            if ln[0] != "G":
                # skip other systems: GLONASS/SBAS use 4 lines, others 8
                i += 4 if ln[0] in "RS" else 8
                continue
            prn = int(ln[1:3])
            y, mo, d, h, mi, s = int(ln[4:8]), int(ln[9:11]), int(ln[12:14]), int(ln[15:17]), int(ln[18:20]), float(ln[21:23])
        else:
            prn = int(ln[0:2])
            yy = int(ln[3:5])
            y = yy + (2000 if yy < 80 else 1900)
            mo, d, h, mi, s = int(ln[6:8]), int(ln[9:11]), int(ln[12:14]), int(ln[15:17]), float(ln[17:22])
        block = lines[i:i + 8]
        i += 8
        vals = [_d(block[0][first_off + k * 19: first_off + (k + 1) * 19]) for k in range(3)]
        for bl in block[1:]:
            bl = bl.rstrip("\n").ljust(80)
            vals += [_d(bl[val_off + k * 19: val_off + (k + 1) * 19]) for k in range(4)]
        rec = dict(zip(_EPH_FIELDS, vals))
        rec["prn"] = prn
        rec["toc"] = np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}", "ns") + np.timedelta64(int(round(s)), "s")
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No GPS ephemerides found in {path}")
    df["toe_gps"] = df["week"] * 604800.0 + df["toe"]
    return df


class BroadcastOrbit:
    """Satellite positions from broadcast ephemerides (IS-GPS-200 algorithm)."""

    def __init__(self, nav_paths):
        if isinstance(nav_paths, (str, bytes)) or hasattr(nav_paths, "__fspath__"):
            nav_paths = [nav_paths]
        self.eph = pd.concat([read_nav(p) for p in nav_paths], ignore_index=True)
        self.eph = self.eph[self.eph["svh"] == 0].drop_duplicates(["prn", "toe_gps"])
        self.prns = set(self.eph["prn"].unique())

    def positions(self, times, prns):
        t = gps_seconds(times)
        prns = np.asarray(prns)
        out = np.full((len(t), 3), np.nan)
        for prn in np.unique(prns):
            sel = np.where(prns == prn)[0]
            e = self.eph[self.eph["prn"] == prn]
            if e.empty:
                continue
            toe_all = e["toe_gps"].to_numpy()
            # nearest ephemeris (by toe) for each epoch
            idx = np.abs(t[sel][:, None] - toe_all[None, :]).argmin(axis=1)
            for k in np.unique(idx):
                rows = sel[idx == k]
                out[rows] = _kepler(e.iloc[k], t[rows])
        return out


def _kepler(eph, t):
    a = eph["sqrta"] ** 2
    n = np.sqrt(GM / a ** 3) + eph["dn"]
    tk = t - eph["toe_gps"]
    m = eph["m0"] + n * tk
    ecc = eph["e"]
    E = m.copy()
    for _ in range(10):
        E = m + ecc * np.sin(E)
    v = np.arctan2(np.sqrt(1 - ecc ** 2) * np.sin(E), np.cos(E) - ecc)
    phi = v + eph["omega"]
    s2, c2 = np.sin(2 * phi), np.cos(2 * phi)
    u = phi + eph["cus"] * s2 + eph["cuc"] * c2
    r = a * (1 - ecc * np.cos(E)) + eph["crs"] * s2 + eph["crc"] * c2
    inc = eph["i0"] + eph["idot"] * tk + eph["cis"] * s2 + eph["cic"] * c2
    x_orb, y_orb = r * np.cos(u), r * np.sin(u)
    toe_sow = eph["toe"]
    om = eph["omega0"] + (eph["omegadot"] - OMEGA_E) * tk - OMEGA_E * toe_sow
    x = x_orb * np.cos(om) - y_orb * np.cos(inc) * np.sin(om)
    y = x_orb * np.sin(om) + y_orb * np.cos(inc) * np.cos(om)
    z = y_orb * np.sin(inc)
    return np.column_stack([x, y, z])


# ─────────────────────────────── SP3 ───────────────────────────────
class SP3Orbit:
    """Precise orbits from one or more SP3 files, Lagrange-interpolated (order 9)."""

    def __init__(self, sp3_paths):
        if isinstance(sp3_paths, (str, bytes)) or hasattr(sp3_paths, "__fspath__"):
            sp3_paths = [sp3_paths]
        recs = []
        for p in sp3_paths:
            with _open(p) as fh:
                t = None
                for ln in fh:
                    if ln.startswith("*"):
                        parts = ln[1:].split()
                        y, mo, d, h, mi = map(int, parts[:5])
                        s = float(parts[5])
                        t = np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}", "ns") + np.timedelta64(int(round(s * 1e9)), "ns")
                    elif ln.startswith("PG") and t is not None:
                        x, y_, z = float(ln[4:18]), float(ln[18:32]), float(ln[32:46])
                        if abs(x) > 1e-6:
                            recs.append((t, int(ln[2:4]), x * 1e3, y_ * 1e3, z * 1e3))
        self.tab = pd.DataFrame(recs, columns=["time", "prn", "x", "y", "z"]).drop_duplicates(["time", "prn"])
        self.prns = set(self.tab["prn"].unique())

    def positions(self, times, prns):
        return _interp_table(self.tab, times, prns)


def _interp_table(tab, times, prns, order=9):
    t = gps_seconds(times)
    prns = np.asarray(prns)
    out = np.full((len(t), 3), np.nan)
    for prn in np.unique(prns):
        sel = np.where(prns == prn)[0]
        g = tab[tab["prn"] == prn].sort_values("time")
        if len(g) < order + 1:
            continue
        tt = gps_seconds(g["time"].to_numpy())
        xyz = g[["x", "y", "z"]].to_numpy()
        pos = np.searchsorted(tt, t[sel])
        start = np.clip(pos - (order + 1) // 2, 0, len(tt) - order - 1)
        for s0 in np.unique(start):
            rows = sel[start == s0]
            nodes = tt[s0:s0 + order + 1]
            vals = xyz[s0:s0 + order + 1]
            x = t[rows]
            # Lagrange basis
            L = np.ones((len(x), order + 1))
            for j in range(order + 1):
                for m in range(order + 1):
                    if m != j:
                        L[:, j] *= (x - nodes[m]) / (nodes[j] - nodes[m])
            out[rows] = L @ vals
        # no extrapolation beyond the table
        bad = (t[sel] < tt[0] - 900) | (t[sel] > tt[-1] + 900)
        out[sel[bad]] = np.nan
    return out


class TableOrbit:
    """Positions from a pre-computed table (columns time, prn, x, y, z)."""

    def __init__(self, table):
        self.tab = table
        self.prns = set(table["prn"].unique())

    def positions(self, times, prns):
        return _interp_table(self.tab, times, prns, order=7)
