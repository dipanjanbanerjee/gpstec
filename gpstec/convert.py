"""Receiver .GPS (NovAtel binary) -> RINEX, using RTKLIB's convbin."""
from pathlib import Path
import shutil
import subprocess

import numpy as np


def find_convbin(hint="convbin"):
    local = Path.home() / ".local" / "bin" / "convbin"
    for cand in (hint, "convbin", "convbin.exe", str(local)):
        p = shutil.which(cand) or (cand if Path(cand).is_file() else None)
        if p:
            return p
    raise FileNotFoundError(
        "RTKLIB convbin was not found. Run `gpstec install-convbin`, or install RTKLIB yourself "
        "and set `convbin = \"/path/to/convbin\"` under [processing] in your config file.")


def gps_to_rinex(gps_file, out_dir, station, settings, freqs=None):
    """Convert one receiver file to RINEX 2.11 obs + nav.

    The station position from the config is written into the header
    (APPROX POSITION XYZ), so the files are also safe to use in other programs.
    Returns (obs_path, nav_path).
    """
    gps_file = Path(gps_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = find_convbin(settings.convbin)
    x, y, z = station.ecef
    stem = gps_file.stem
    obs = out_dir / f"{stem}.obs"
    nav = out_dir / f"{stem}.nav"
    freqs = freqs or (3 if settings.pair == "L1L5" else 2)
    cmd = [exe, "-r", settings.convbin_format, "-v", "2.11", "-f", str(freqs),
           "-od", "-os", "-y", "R", "-y", "E", "-y", "J", "-y", "S", "-y", "C",
           "-hm", station.name, "-hp", f"{x:.4f}/{y:.4f}/{z:.4f}",
           "-o", str(obs), "-n", str(nav), str(gps_file)]
    print("  " + " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not obs.exists():
        raise RuntimeError(f"convbin failed for {gps_file}:\n{res.stdout}\n{res.stderr}")
    obs = rinex2_rename(obs, station.name)
    if nav.exists():
        new_nav = obs.with_name(obs.name[:-1] + "n")
        if new_nav.exists():
            new_nav.unlink()
        nav = nav.rename(new_nav)
    else:
        nav = None
    return obs, nav


def rinex2_rename(path, site, kind="o"):
    """Rename to the RINEX 2 short name ssssDDD0.YYo (what Gopi's software expects)."""
    path = Path(path)
    first = _first_obs_time(path)
    if first is None:
        return path
    doy = (first.astype("datetime64[D]") - first.astype("datetime64[Y]")).astype(int) + 1
    yy = first.astype("datetime64[Y]").astype(int) + 1970
    new = path.with_name(f"{site[:4].lower():<4}{doy:03d}0.{yy % 100:02d}{kind}")
    if new.exists():
        new.unlink()
    path.rename(new)
    return new


def _first_obs_time(path):
    with open(path, errors="replace") as fh:
        for ln in fh:
            if "TIME OF FIRST OBS" in ln:
                y, mo, d = ln[:6].strip(), ln[6:12].strip(), ln[12:18].strip()
                return np.datetime64(f"{int(y):04d}-{int(mo):02d}-{int(d):02d}")
            if "END OF HEADER" in ln:
                return None
