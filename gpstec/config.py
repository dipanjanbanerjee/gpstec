"""Station and processing settings, loaded from a TOML file."""
from dataclasses import dataclass, field, fields
from pathlib import Path
try:
    import tomllib
except ImportError:          # Python < 3.11
    import tomli as tomllib

import numpy as np

from .geometry import geodetic_to_ecef


@dataclass
class Station:
    name: str = "stat"
    lat: float = None             # degrees North   (required: the surveyed antenna position)
    lon: float = None             # degrees East
    height: float = 0.0           # metres above the WGS84 ellipsoid
    utc_offset_h: float = 0.0     # for the local-time axis on plots (e.g. 5.5 for IST)

    @property
    def ecef(self):
        return geodetic_to_ecef(self.lat, self.lon, self.height)


@dataclass
class Settings:
    # geometry
    shell_height_km: float = 350.0
    # elevation limits (degrees)
    mask_elevation: float = 10.0        # ignore anything lower, everywhere
    level_elevation: float = 20.0       # phase-to-code levelling uses points above this
    calib_elevation: float = 30.0       # receiver-bias estimation uses points above this
    calib_night_lt: list = field(default_factory=lambda: [22.0, 6.0])  # local-time window for the
                                        # receiver bias (quiet night: weak gradients); [] = whole day
    output_elevation: float = 50.0      # VTEC products use points above this
    # arcs
    max_gap_s: float = 60.0
    slip_threshold_m: float = 0.25
    min_arc_s: float = 600.0
    # signals
    pair: str = "L1L2"
    # biases
    sat_dcb: str = "auto"               # "auto" | "file" | "estimate"
    dcb_files: list = field(default_factory=list)   # CODE P1P2*.DCB / P1C1*.DCB (or .BSX)
    dcb_dir: str = "dcb"                # cache for downloaded CODE files
    receiver_bias: str = "lsq"          # "lsq" | "minstd" | "gim" | number in TECU
    lsq_elevation: float = 20.0         # elevation cut-off for the least-squares receiver bias
    receiver_bias_multiday: str = "per_day"  # "per_day" (each day on its own) | "median" over the run
    enforce_nonnegative: bool = True    # raise the receiver bias if it would make VTEC negative
    floor_percentile: float = 99.5      # ...ignoring the noisiest 0.5 % of points
    gim: str = "auto"                   # CODE global ionosphere map as a reference: "auto" | "off"
    gim_dir: str = "gim"
    gim_station: str = "bilinear"       # GIM value at the station: "bilinear" (4 grid nodes) | "nearest" node
    # daily products
    bin_minutes: float = 1.0
    clip_sigma: float = 2.0             # outlier clipping inside each bin
    min_sats_per_bin: int = 1
    smooth_curve: bool = True           # gradient-corrected least-squares fit of all satellites (main curve)
    smooth_knot_min: float = 60.0       # spacing of that spline curve (minutes)
    # figures
    figure_formats: list = field(default_factory=lambda: ["png", "pdf"])  # also "svg", "eps", "tiff"
    figure_dpi: int = 300               # raster resolution (PNG/TIFF)
    figure_width_in: float = 7.2        # 7.2 in = full page width in most journals; 3.5 = one column
    figure_titles: bool = True          # False: no title line (the caption goes in the paper)
    # tools
    convbin: str = "convbin"            # path to RTKLIB convbin(.exe)
    convbin_format: str = "nov"         # NovAtel OEM6/OEM7 (GPStation-6)


def load(path=None):
    """Return (Station, Settings). Missing keys keep their defaults."""
    st, se = Station(), Settings()
    if path:
        with open(path, "rb") as fh:
            cfg = tomllib.load(fh)
        for obj, sec in ((st, "station"), (se, "processing")):
            known = {f.name for f in fields(obj)}
            for key, val in cfg.get(sec, {}).items():
                if key not in known:
                    raise KeyError(f"Unknown setting [{sec}] {key} in {path}")
                setattr(obj, key, val)
    if st.lat is None or st.lon is None:
        raise ValueError("The station position is not set. Run `gpstec init`, put your antenna's "
                         "lat/lon/height in the [station] block of station.toml, and pass it with -c.")
    return st, se
