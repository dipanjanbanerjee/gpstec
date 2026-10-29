"""Satellite differential code biases (DCBs): finding, downloading and reading them.

What the TEC processing needs, per GPS satellite, is the bias of the code pair it
used, in nanoseconds:

    receiver gives P1 and P2 -> DCB(P1-P2)          = C1W - C2W
    receiver gives C1 and P2 -> DCB(C1-P2)          = C1C - C2W   (the GPStation-6 case)

Supported file types (any of these is enough):

  * CODE monthly .DCB files        P1P2YYMM.DCB (+ P1C1YYMM.DCB for C1 receivers)
  * Bias-SINEX DSB files (.BSX)    e.g. CAS0MGXRAP_YYYYDDD0000_01D_01D_DCB.BSX (Chinese
                                   Academy of Sciences, daily, all GNSS)
  * Bias-SINEX OSB files (.BIA)    e.g. CODE / CAS observable-specific biases

``find_or_fetch`` first looks in the local dcb folder, then tries several public
servers in turn. Files you download yourself by browser can simply be dropped into
the dcb folder; they are picked up by date.
"""
from pathlib import Path
import gzip
import re
import urllib.request

import numpy as np

from .constants import TECU_PER_NS_L1L2


# ───────────────────────────── sources ─────────────────────────────
def _cas_name(y, doy):
    return f"CAS0MGXRAP_{y}{doy:03d}0000_01D_01D_DCB.BSX.gz"


def candidate_urls(day):
    """(local file name, url) pairs to try, best first."""
    day = np.datetime64(day, "D").astype(object)
    y, yy, mm = day.year, day.year % 100, day.month
    doy = day.timetuple().tm_yday
    cas = _cas_name(y, doy)
    out = []
    # 1. CODE monthly DCBs (http and ftp)
    for proto in ("http://www.aiub.unibe.ch/download", "http://ftp.aiub.unibe.ch", "ftp://ftp.aiub.unibe.ch"):
        out.append((f"P1P2{yy:02d}{mm:02d}.DCB.Z", f"{proto}/CODE/{y}/P1P2{yy:02d}{mm:02d}.DCB.Z"))
        out.append((f"P1C1{yy:02d}{mm:02d}.DCB.Z", f"{proto}/CODE/{y}/P1C1{yy:02d}{mm:02d}.DCB.Z"))
    # 2. CAS daily multi-GNSS DCBs on several mirrors
    for url in (
        f"https://igs.ign.fr/pub/igs/products/mgex/dcb/{y}/{cas}",
        f"ftp://igs.ign.fr/pub/igs/products/mgex/dcb/{y}/{cas}",
        f"ftp://igs.gnsswhu.cn/pub/gps/products/mgex/dcb/{y}/{cas}",
        f"ftp://ftp.gipp.org.cn/product/dcb/mgex/{y}/{cas}",
        f"https://igs.bkg.bund.de/root_ftp/IGS/products/mgex/dcb/{y}/{cas}",
    ):
        out.append((cas, url))
    return out


def manual_download_help(day):
    day = np.datetime64(day, "D").astype(object)
    y, yy, mm = day.year, day.year % 100, day.month
    cas = _cas_name(y, day.timetuple().tm_yday)
    return (
        "Download ONE of these in a browser and put it in the dcb folder (no need to unpack):\n"
        f"  http://www.aiub.unibe.ch/download/CODE/{y}/P1P2{yy:02d}{mm:02d}.DCB.Z  and  P1C1{yy:02d}{mm:02d}.DCB.Z\n"
        f"  https://cddis.nasa.gov/archive/gnss/products/bias/{y}/{cas}   (free NASA Earthdata login)\n"
        f"  https://igs.ign.fr/pub/igs/products/mgex/dcb/{y}/{cas}")


def find_or_fetch(day, cache_dir, timeout=20, log=print, offline=False):
    """Return a list of usable local bias files for ``day`` (may be empty)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    local = local_files_for(day, cache)
    if local:
        return local
    if offline:
        return []
    got = []
    have_p1p2 = False
    for name, url in candidate_urls(day):
        if name.startswith("P1C1") and not have_p1p2:
            continue                       # only worth having together with P1P2
        if name.startswith("P1P2") and have_p1p2:
            continue
        if name.startswith("CAS") and have_p1p2:
            break
        target = cache / name
        if target.exists():
            got.append(target)
            continue
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            target.write_bytes(data)
            log(f"  downloaded {url}")
            got.append(target)
            if name.startswith("P1P2"):
                have_p1p2 = True
            if name.startswith("CAS"):
                break
        except Exception as e:
            log(f"  not reachable: {url} ({type(e).__name__})")
    return [p for p in got if p.exists()]


def local_files_for(day, folder):
    """Bias files already in ``folder`` that cover ``day``."""
    day_ = np.datetime64(day, "D").astype(object)
    y, yy, mm = day_.year, day_.year % 100, day_.month
    doy = day_.timetuple().tm_yday
    out = []
    for p in sorted(Path(folder).glob("*")):
        n = p.name.upper()
        if n.startswith(f"P1P2{yy:02d}{mm:02d}.DCB") or n.startswith(f"P1C1{yy:02d}{mm:02d}.DCB"):
            out.append(p)
        elif re.search(rf"_{y}{doy:03d}0000_01D_", n) and (".BSX" in n or ".BIA" in n):
            out.append(p)
    return out


# ───────────────────────────── readers ─────────────────────────────
def _read_text(path):
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x9d":                     # unix compress (.Z)
        raw = _unlzw(raw)
    elif raw[:2] == b"\x1f\x8b":                   # gzip
        raw = gzip.decompress(raw)
    return raw.decode("ascii", errors="replace")


def _unlzw(data):
    try:
        import unlzw3
    except ImportError as e:
        raise RuntimeError("Install 'unlzw3' (pip install unlzw3) to unpack .Z files") from e
    return unlzw3.unlzw(data)


def read_code_dcb(path):
    """CODE .DCB file -> {prn: bias_ns}."""
    out = {}
    for ln in _read_text(path).splitlines():
        m = re.match(r"^G(\d\d)\s+(?:\S+\s+)?(-?\d+\.\d+)\s+(-?\d+\.\d+)", ln)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def read_bias_sinex(path, day=None):
    """Bias-SINEX -> {"DSB": {(prn, o1, o2): ns}, "OSB": {(prn, o): ns}} for GPS satellites."""
    dsb, osb = {}, {}
    in_sol = False
    for ln in _read_text(path).splitlines():
        if ln.startswith("+BIAS/SOLUTION"):
            in_sol = True
            continue
        if ln.startswith("-BIAS/SOLUTION"):
            in_sol = False
            continue
        if not in_sol or ln.startswith("*"):
            continue
        kind = ln[1:4].strip()
        prn = ln[11:14].strip()
        station = ln[15:24].strip()
        if not prn.startswith("G") or station:
            continue
        o1, o2 = ln[25:29].strip(), ln[30:34].strip()
        unit = ln[65:69].strip()
        try:
            val = float(ln[70:91])
        except ValueError:
            continue
        if unit != "ns":
            continue
        p = int(prn[1:])
        if kind == "DSB":
            dsb[(p, o1, o2)] = val
        elif kind == "OSB":
            osb[(p, o1)] = val
    return {"DSB": dsb, "OSB": osb}


def _pair_from_sinex(tab, prn, a, b):
    """Bias a - b (ns) from DSB or OSB entries, trying direct, reversed and chained forms."""
    d, o = tab["DSB"], tab["OSB"]
    if (prn, a, b) in d:
        return d[(prn, a, b)]
    if (prn, b, a) in d:
        return -d[(prn, b, a)]
    if (prn, a) in o and (prn, b) in o:
        return o[(prn, a)] - o[(prn, b)]
    for mid in ("C1W", "C2W", "C1C"):          # a - mid + mid - b
        if mid in (a, b):
            continue
        x = (prn, a, mid) in d or (prn, mid, a) in d
        y = (prn, mid, b) in d or (prn, b, mid) in d
        if x and y:
            return _pair_from_sinex(tab, prn, a, mid) + _pair_from_sinex(tab, prn, mid, b)
    return None


def satellite_bias_tecu(paths, code_used):
    """Satellite bias in TECU to ADD to the leveled STEC, per PRN.

    TEC = K (P2 - X1) + K c DCB(X1 - P2), with X1 = P1 or C1 the code actually used.
    """
    first = "C1C" if code_used == "C1" else "C1W"
    p1p2, p1c1, ns = {}, {}, {}
    for p in paths:
        n = Path(p).name.upper()
        if n.startswith("P1P2"):
            p1p2.update(read_code_dcb(p))
        elif n.startswith("P1C1"):
            p1c1.update(read_code_dcb(p))
        else:
            tab = read_bias_sinex(p)
            for prn in range(1, 33):
                v = _pair_from_sinex(tab, prn, first, "C2W")
                if v is not None:
                    ns.setdefault(prn, v)
    if p1p2 and not ns:
        if code_used == "C1" and not p1c1:
            print("  warning: data uses C1 but no P1C1 file was found; P1-C1 biases ignored")
        for prn, b in p1p2.items():
            ns[prn] = b - (p1c1.get(prn, 0.0) if code_used == "C1" else 0.0)
    return {prn: v * TECU_PER_NS_L1L2 for prn, v in ns.items()}
