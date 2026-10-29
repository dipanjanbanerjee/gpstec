import numpy as np
import pandas as pd
import pytest

from gpstec.config import Settings, Station


@pytest.fixture
def station():
    return Station(name="test", lat=23.36, lon=86.34, height=285.0, utc_offset_h=5.5)


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.figure_formats = ["png", "pdf"]
    return s


def synthetic_day(day="2022-12-25", n_sat=6, sta=(23.36, 86.34)):
    """Per-satellite VTEC tracks with a diurnal curve and a N-S gradient."""
    t0 = np.datetime64(day, "ns")
    rows = []
    rng = np.random.default_rng(1)
    for k in range(n_sat):
        start = k * 4 * 3600
        secs = np.arange(start, min(start + 6 * 3600, 86400), 30)
        h = secs / 3600.0
        phase = (h - start / 3600) / 6
        el = 20 + 65 * np.sin(np.pi * phase)
        dlat = 3 * np.cos(np.pi * phase)
        dlon = 2 * np.sin(2 * np.pi * phase)
        base = 5 + 65 * np.exp(-((h - 8) / 2.5) ** 2)
        v = base - 4 * dlat + rng.normal(0, 0.2, len(h))
        rows.append(pd.DataFrame({"time": t0 + (secs * 1e9).astype("timedelta64[ns]"), "prn": k + 1,
                                  "el": el, "ipp_lat": sta[0] + dlat, "ipp_lon": sta[1] + dlon,
                                  "vtec": v, "stec": v / 0.9}))
    return pd.concat(rows, ignore_index=True)
