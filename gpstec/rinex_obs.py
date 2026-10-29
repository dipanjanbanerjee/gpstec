"""RINEX observation file reader (versions 2.x and 3.x), GPS satellites only.

Returns a tidy table with one row per (epoch, satellite) and the columns the TEC
processing needs:

    time   numpy datetime64 (receiver time tag, GPS time)
    prn    int
    C1 P1 P2 C2 C5   pseudoranges (m), NaN when missing
    L1 L2 L5         carrier phases (cycles)
    LLI1 LLI2 LLI5   loss-of-lock indicators (0 = OK)
    S1 S2            signal strength (dB-Hz), when present

RINEX 3 codes are mapped to these names: C1C->C1, C1W/C1P->P1, C2W/C2P->P2,
C2L/C2S/C2X->C2, L1C->L1, L2W/L2P (or L2L/L2X if no W)->L2, C5Q/C5X->C5, L5Q/L5X->L5.
"""
from dataclasses import dataclass, field
import gzip
import numpy as np
import pandas as pd


@dataclass
class ObsHeader:
    version: float = 2.11
    marker: str = ""
    approx_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    obs_types: list = field(default_factory=list)   # v2: list; v3: GPS list
    interval: float = np.nan
    first_obs: str = ""
    comments: list = field(default_factory=list)


_V3_MAP = {
    "C1C": "C1", "C1W": "P1", "C1P": "P1",
    "C2W": "P2", "C2P": "P2", "C2L": "C2", "C2S": "C2", "C2X": "C2",
    "C5Q": "C5", "C5X": "C5", "C5I": "C5",
    "L1C": "L1", "L2W": "L2", "L2P": "L2", "L5Q": "L5", "L5X": "L5", "L5I": "L5",
    "S1C": "S1", "S2W": "S2", "S2P": "S2",
}
_V3_L2_FALLBACK = ("L2L", "L2X", "L2S")

_KEEP = ["C1", "P1", "P2", "C2", "C5", "L1", "L2", "L5", "S1", "S2"]


def _open(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


def _f(s):
    s = s.strip()
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def read_header(fh):
    hdr = ObsHeader()
    v3_gps_types = []
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("Unexpected end of file inside the RINEX header")
        label = line[60:80].strip()
        if label == "RINEX VERSION / TYPE":
            hdr.version = float(line[:9])
        elif label == "MARKER NAME":
            hdr.marker = line[:60].strip()
        elif label == "APPROX POSITION XYZ":
            hdr.approx_xyz = np.array([_f(line[0:14]), _f(line[14:28]), _f(line[28:42])])
        elif label == "# / TYPES OF OBSERV":            # RINEX 2
            if line[:6].strip():
                hdr.obs_types = []
            hdr.obs_types += line[6:60].split()
        elif label == "SYS / # / OBS TYPES":            # RINEX 3
            if line[0] == "G":
                v3_gps_types = line[7:60].split()
                cur = "G"
            elif line[0] == " " and v3_gps_types and cur == "G":
                v3_gps_types += line[7:60].split()
            else:
                cur = line[0]
        elif label == "INTERVAL":
            hdr.interval = _f(line[:10])
        elif label == "TIME OF FIRST OBS":
            hdr.first_obs = line[:43]
        elif label == "COMMENT":
            hdr.comments.append(line[:60].rstrip())
        elif label == "END OF HEADER":
            break
    if hdr.version >= 3:
        hdr.obs_types = v3_gps_types
    return hdr


def _column_plan(hdr):
    """Map each observation slot index to an output column name (or None)."""
    if hdr.version < 3:
        return [t if t in _KEEP else None for t in hdr.obs_types]
    plan = [_V3_MAP.get(t) for t in hdr.obs_types]
    if "L2" not in plan:
        for alt in _V3_L2_FALLBACK:
            if alt in hdr.obs_types:
                plan[hdr.obs_types.index(alt)] = "L2"
                break
    # keep the first slot when two codes map to the same name
    seen = set()
    for i, name in enumerate(plan):
        if name in seen:
            plan[i] = None
        elif name:
            seen.add(name)
    return plan


def read_obs(path, gps_only=True):
    """Read a RINEX observation file. Returns (header, DataFrame)."""
    with _open(path) as fh:
        hdr = read_header(fh)
        plan = _column_plan(hdr)
        nobs = len(plan)
        cols = {name: [] for name in _KEEP}
        lli = {"L1": [], "L2": [], "L5": []}
        times, prns = [], []

        def add_obs(t, prn, rec):
            # rec is the concatenated observation text for one satellite (16 chars per obs)
            times.append(t)
            prns.append(prn)
            got = dict.fromkeys(_KEEP, np.nan)
            got_lli = {"L1": 0, "L2": 0, "L5": 0}
            for i, name in enumerate(plan):
                if name is None:
                    continue
                chunk = rec[16 * i:16 * i + 16]
                got[name] = _f(chunk[:14])
                if name in got_lli:
                    c = chunk[14:15]
                    got_lli[name] = int(c) if c.strip().isdigit() else 0
            for name in _KEEP:
                cols[name].append(got[name])
            for name in lli:
                lli[name].append(got_lli[name])

        if hdr.version < 3:
            _read_body_v2(fh, nobs, add_obs, gps_only)
        else:
            _read_body_v3(fh, nobs, add_obs, gps_only)

    df = pd.DataFrame({"time": np.array(times, dtype="datetime64[ns]"), "prn": np.array(prns, dtype=np.int16)})
    for name in _KEEP:
        df[name] = np.array(cols[name], dtype=float)
    for name in lli:
        df["LLI" + name[1]] = np.array(lli[name], dtype=np.int8)
    df = df.dropna(axis=1, how="all")
    return hdr, df


def _epoch_time_v2(line):
    yy = int(line[1:3])
    year = yy + (2000 if yy < 80 else 1900)
    mo, dd, hh, mi = int(line[4:6]), int(line[7:9]), int(line[10:12]), int(line[13:15])
    sec = float(line[15:26])
    base = np.datetime64(f"{year:04d}-{mo:02d}-{dd:02d}T{hh:02d}:{mi:02d}", "ns")
    return base + np.timedelta64(int(round(sec * 1e9)), "ns")


def _read_body_v2(fh, nobs, add_obs, gps_only):
    lines_per_sat = (nobs + 4) // 5
    while True:
        line = fh.readline()
        if not line:
            break
        if len(line) < 32 or not line[1:3].strip():
            continue
        flag = int(line[28]) if line[28].strip() else 0
        nsat = int(line[29:32])
        if flag > 1:                      # event records: skip the following lines
            for _ in range(nsat):
                fh.readline()
            continue
        t = _epoch_time_v2(line)
        sat_ids = []
        cur = line
        while True:
            take = min(12, nsat - len(sat_ids))
            field_ = cur.rstrip("\n").ljust(68)[32:32 + 3 * take]
            sat_ids += [field_[3 * i:3 * i + 3] for i in range(take)]
            if len(sat_ids) >= nsat:
                break
            cur = fh.readline()
        for sid in sat_ids:
            rec = ""
            for _ in range(lines_per_sat):
                ln = fh.readline().rstrip("\n")
                rec += ln.ljust(80)[:80]
            sysch = sid[0] if sid[0] != " " else "G"
            if gps_only and sysch != "G":
                continue
            add_obs(t, int(sid[1:3]), rec)


def _read_body_v3(fh, nobs, add_obs, gps_only):
    while True:
        line = fh.readline()
        if not line:
            break
        if not line.startswith(">"):
            continue
        flag = int(line[31]) if line[31].strip() else 0
        nsat = int(line[32:35])
        if flag > 1:
            for _ in range(nsat):
                fh.readline()
            continue
        year, mo, dd, hh, mi = (int(line[2:6]), int(line[7:9]), int(line[10:12]),
                                int(line[13:15]), int(line[16:18]))
        sec = float(line[18:29])
        t = np.datetime64(f"{year:04d}-{mo:02d}-{dd:02d}T{hh:02d}:{mi:02d}", "ns") + \
            np.timedelta64(int(round(sec * 1e9)), "ns")
        for _ in range(nsat):
            ln = fh.readline().rstrip("\n")
            sid = ln[:3]
            if gps_only and sid[0] != "G":
                continue
            add_obs(t, int(sid[1:3]), ln[3:].ljust(16 * nobs))
