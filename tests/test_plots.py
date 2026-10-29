import numpy as np
import pandas as pd

from gpstec import plots, products
from conftest import synthetic_day


def _series(d, day, station, settings):
    s = products.daily_series(d, day, el_min=settings.output_elevation, bin_min=1)
    sm = products.smooth_station_vtec(d, day, station.lat, station.lon, el_min=settings.output_elevation)
    for c in ("vtec_smooth", "grad_north", "grad_east"):
        s[c] = sm[c].to_numpy()
    s["vtec_uncertainty"] = 1.2
    s["vtec_gim"] = s["vtec_smooth"] * 0.85 + 2
    return s


def test_figures_have_no_overlapping_text(tmp_path, station, settings):
    days = [str(np.datetime64("2022-12-25") + np.timedelta64(k, "D")) for k in range(7)]
    series = []
    for day in days:
        d = synthetic_day(day)
        s = _series(d, day, station, settings)
        series.append(s)
        figs = [plots.plot_day(d, s, station, settings, None, return_fig=True),
                plots.plot_ipp_map(d, station, settings, None, day, return_fig=True)]
        for f in figs:
            assert plots.text_overlaps(f) == []
    allv = pd.concat(series, ignore_index=True)
    rows = [{"next_date": days[k], "jump_median_el30": [0.8, 2.5, -0.6][k % 3], "tolerance_tecu": 1.8,
             "status": "OK" if k % 3 != 1 else "CHECK"} for k in range(1, 7)]
    for f in (plots.plot_overlay(allv, station, settings, None, return_fig=True),
              plots.plot_heatmap(allv, station, settings, None, return_fig=True),
              plots.plot_continuity(allv, rows, station, settings, None, return_fig=True)):
        assert plots.text_overlaps(f) == []


def test_save_writes_every_format(tmp_path, station, settings):
    day = "2022-12-25"
    d = synthetic_day(day)
    s = _series(d, day, station, settings)
    out = plots.plot_day(d, s, station, settings, tmp_path / "x_vtec.png")
    assert sorted(p.suffix for p in out) == [".pdf", ".png"]
    assert all(p.stat().st_size > 10_000 for p in out)
