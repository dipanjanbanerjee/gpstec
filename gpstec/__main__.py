"""Command line interface.

  gpstec init                       write a station.toml template to edit
  gpstec run -c station.toml -o results FILES...
  gpstec events results [--flares flares.csv] [--dst kyoto]   flare and storm signatures
  gpstec install-convbin            build RTKLIB convbin (needed for .GPS input)
  gpstec --version
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path

from . import __author__, __copyright__, __version__

TEMPLATE = "station_template.toml"


def _init(path, force=False):
    path = Path(path)
    if path.exists() and not force:
        print(f"{path} already exists (use --force to overwrite).")
        return 1
    text = resources.files("gpstec").joinpath("data", TEMPLATE).read_text(encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}. Edit the [station] block (name, lat, lon, height, utc_offset_h), then run:")
    print(f"  gpstec run -c {path} -o results  your_file.GPS")
    return 0


def _install_convbin(dest):
    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    for tool in ("git", "make", "gcc"):
        if not shutil.which(tool):
            print(f"'{tool}' is needed to build convbin. On Ubuntu/Debian: sudo apt install git build-essential")
            return 1
    src = Path.home() / ".cache" / "gpstec-rtklib"
    if src.exists():
        shutil.rmtree(src)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/rtklibexplorer/RTKLIB.git", str(src)],
                   check=True)
    build = src / "app" / "consapp" / "convbin" / "gcc"
    subprocess.run(["make", "-C", str(build)], check=True)
    shutil.copy2(build / "convbin", dest / "convbin")
    print(f"convbin installed to {dest / 'convbin'}")
    if str(dest) not in os.environ.get("PATH", "").split(os.pathsep):
        print(f"Make sure {dest} is on your PATH, or set  convbin = \"{dest / 'convbin'}\"  in station.toml.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="gpstec",
        description="Calibrated GPS total electron content (TEC) from NovAtel GPStation-6 and other "
                    "dual-frequency receivers.",
        epilog=f"{__copyright__}. https://github.com/dipanjanbanerjee/gpstec",
        formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-V", "--version", action="version",
                    version=f"gpstec {__version__}\n{__copyright__}\nAuthor: {__author__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="write a station.toml configuration template")
    i.add_argument("path", nargs="?", default="station.toml")
    i.add_argument("--force", action="store_true", help="overwrite an existing file")

    c = sub.add_parser("install-convbin", help="download and build RTKLIB convbin (Linux/macOS)")
    c.add_argument("--dest", default="~/.local/bin", help="where to put convbin (default ~/.local/bin)")

    r = sub.add_parser("run", help="process .GPS receiver files or RINEX files")
    r.add_argument("files", nargs="+", help=".GPS receiver files or RINEX observation files (wildcards OK)")
    r.add_argument("-c", "--config", help="TOML config file (station position, settings)")
    r.add_argument("-o", "--out", default="tec_output", help="output folder")
    r.add_argument("--nav", help="navigation file (if not next to the observation file)")
    r.add_argument("--sp3", nargs="*", help="precise orbit SP3 file(s) instead of broadcast")
    r.add_argument("--dcb", nargs="*", help="CODE P1P2/P1C1 .DCB or Bias-SINEX files")
    r.add_argument("--elev", type=float, help="elevation cut for VTEC output (default 50)")
    r.add_argument("--rx-bias", help="receiver bias: lsq, minstd, gim or a number in TECU")
    r.add_argument("--no-cmn", action="store_true", help="skip writing the large .Cmn-style file")
    e = sub.add_parser("events", help="solar flare and geomagnetic storm signatures in the output of `run`")
    e.add_argument("results", help="output folder written by `gpstec run -o`")
    e.add_argument("-c", "--config", help="station.toml (optional; the station is read from the .Cmn files)")
    e.add_argument("--flares", help="CSV of flares: class,start,peak,end (UTC)")
    e.add_argument("--dst", help="'kyoto' to download hourly Dst from WDC Kyoto, or a Dst file "
                                 "(Kyoto WDC format, saved Kyoto monthly page, IAGA-2002 or time,dst CSV)")
    e.add_argument("--quiet-days", help="comma-separated quiet days YYYY-MM-DD for the reference "
                                        "(default: days with Dst >= --quiet-dst all day)")
    e.add_argument("--quiet-dst", type=float, default=-30.0, help="quiet-day Dst threshold in nT (default -30)")
    e.add_argument("--window", type=int, default=7, help="use quiet days within +-N days (default 7)")
    e.add_argument("--elev", type=float, default=30.0,
                   help="elevation cut for the flare analysis (default 30; the storm part uses the VTEC from `run`)")
    e.add_argument("-o", "--out", help="output folder (default RESULTS/events)")
    a = ap.parse_args(argv)

    if a.cmd == "init":
        return _init(a.path, a.force)
    if a.cmd == "install-convbin":
        return _install_convbin(a.dest)

    if a.cmd == "events":
        from . import config, events
        if a.config:
            station, settings = config.load(a.config)
        else:
            station, settings = None, config.Settings()
        el = a.elev
        q = [d.strip() for d in a.quiet_days.split(",")] if a.quiet_days else None
        events.run(a.results, flares_csv=a.flares, dst=a.dst, quiet_days=q, quiet_dst=a.quiet_dst,
                   window_days=a.window, el_min=el, station=station,
                   out_dir=a.out, settings=settings)
        return 0

    from . import config, pipeline          # heavy imports only when processing
    print(f"gpstec {__version__}  ·  {__copyright__}")
    station, settings = config.load(a.config)
    if a.elev is not None:
        settings.output_elevation = a.elev
    if a.dcb:
        settings.dcb_files = a.dcb
        settings.sat_dcb = "file"
    if a.rx_bias:
        try:
            settings.receiver_bias = float(a.rx_bias)
        except ValueError:
            settings.receiver_bias = a.rx_bias
    files = []
    for f in a.files:
        files += sorted(glob.glob(f)) or [f]
    pipeline.run([Path(f) for f in files], station, settings, a.out, nav=a.nav, sp3=a.sp3,
                 write_cmn=not a.no_cmn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
