import numpy as np
import pandas as pd

from gpstec import events, plots, products
from gpstec.config import Settings, Station

KYOTO_PAGE = """<pre>
                                           NOVEMBER   2025
      unit=nT                                                                                      UT
      1   2   3   4   5   6   7   8    9  10  11  12  13  14  15  16   17  18  19  20  21  22  23  24
DAY
11  -13 -15 -16 -16 -13 -10 -10 -10  -10  -7  -4  -5  -3  -1   2   5    9  10   9  11  14  15  17  34
12   51 -65-155-159-217-175-189-186 -175-209-189-208-187-169-169-155 -133-121-112 -82 -59 -63 -72 -72
</pre>"""


def test_parse_kyoto_page():
    d = events.parse_dst(KYOTO_PAGE)
    assert len(d) == 48
    assert d["dst"].min() == -217
    assert d.loc[d["dst"].idxmin(), "time"] == pd.Timestamp("2025-11-12 04:30")


def test_parse_wdc_format():
    vals = [-13, -15, -16, -16, -13, -10, -10, -10, -10, -7, -4, -5, -3, -1, 2, 5, 9, 10, 9, 11, 14, 15, 17, 34]
    line = "DST2511*11PPX020   0" + "".join(f"{v:4d}" for v in vals) + f"{-1:4d}"
    d = events.parse_dst(line)
    assert len(d) == 24 and d["dst"].iloc[-1] == 34 and d["time"].iloc[0] == pd.Timestamp("2025-11-11 00:30")


def test_solar_zenith_noon():
    sza, lst = events.solar_zenith("2025-03-20 06:15", 0.0, 86.3)
    assert sza < 3 and abs(lst - 12) < 0.3
    sza, _ = events.solar_zenith("2025-11-11 18:00", 23.36, 86.34)
    assert sza > 95


def _synthetic(days, flare_peak=None, amp=3.0, rise_min=15, seed=3):
    """1-s VTEC of 6 satellites per day (03-11 UT) with wave-like fluctuations, optional flare."""
    rng = np.random.default_rng(seed)
    out = []
    for day in days:
        t0 = pd.Timestamp(day)
        for k in range(6):
            a = 3 * 3600 + k * 1800
            secs = np.arange(a, a + 5 * 3600)
            h = secs / 3600
            el = 35 + 50 * np.sin(np.pi * (secs - a) / (5 * 3600))
            v = 10 + 50 * np.exp(-((h - 7) / 3) ** 2) + 0.4 * k
            for _ in range(3):             # travelling-disturbance-like waves, different per satellite
                v += rng.uniform(0.1, 0.3) * np.sin(2 * np.pi * secs / rng.uniform(1200, 3600) + rng.uniform(0, 6))
            v += rng.normal(0, 0.03, secs.size)
            t = t0 + pd.to_timedelta(secs, unit="s")
            if flare_peak is not None:
                tm = (t - (flare_peak - pd.Timedelta(minutes=rise_min))).total_seconds().to_numpy() / 60
                v += np.where(tm < 0, 0, np.where(tm < rise_min, amp * tm / rise_min,
                                                  amp * np.exp(-(tm - rise_min) / 30)))
            out.append(pd.DataFrame({"time": t, "prn": k + 1, "az": 100.0, "el": el, "ipp_lat": 23.0,
                                     "ipp_lon": 86.0, "stec": v / np.sin(np.radians(el)), "vtec": v}))
    return pd.concat(out, ignore_index=True)


def test_flare_detected_and_null_quiet():
    days = [f"2025-11-{d:02d}" for d in range(1, 8)]
    peak = pd.Timestamp("2025-11-04 07:00")
    d = _synthetic(days, flare_peak=peak)
    p30 = {day: events.passes_30s(d[d["time"].dt.strftime("%Y-%m-%d") == day], 30) for day in days}
    allp = pd.concat(p30.values(), ignore_index=True)
    exc, n, per = events.flare_excess(allp, peak - pd.Timedelta(minutes=15), peak)
    null = events.null_excesses(p30, peak - pd.Timedelta(minutes=15), peak, [peak])
    st = events.flare_statistics(exc, n, null, per)
    assert 2.0 < exc < 3.5 and n >= 4
    assert st["detected"] and st["p_value"] < 0.01
    # the same test at an ordinary time is not a detection
    fake = pd.Timestamp("2025-11-06 07:00")
    exc0, n0, per0 = events.flare_excess(allp, fake - pd.Timedelta(minutes=15), fake)
    null0 = events.null_excesses(p30, fake - pd.Timedelta(minutes=15), fake, [peak, fake])
    assert not events.flare_statistics(exc0, n0, null0, per0)["detected"]


def test_storm_deviation_finds_depletion():
    rows = []
    for k in range(15):
        day = f"2025-11-{k + 1:02d}"
        h = np.arange(1440) / 60
        v = 5 + 60 * np.exp(-((h - 7) / 3) ** 2) + 0.5 * np.sin(k)
        if day == "2025-11-12":
            v = v * 0.6
        rows.append(pd.DataFrame({"date": day, "ut_hour": h, "vtec": v, "vtec_smooth": v,
                                  "vtec_gim": np.where(np.arange(1440) % 60 == 0, v * 0.9, np.nan)}))
    allv = pd.concat(rows, ignore_index=True)
    quiet = [f"2025-11-{k:02d}" for k in range(1, 16) if k != 12]
    dev = events.storm_deviation(allv, quiet)
    g = dev[dev["date"] == "2025-11-12"]
    assert g["dvtec_percent"].min() < -35 and (g["vtec"] < g["lower_bound"]).sum() > 300
    assert (dev[dev["date"] == "2025-11-05"]["outside_bounds"]).sum() == 0
    dst = pd.DataFrame({"time": pd.date_range("2025-11-01 00:30", periods=15 * 24, freq="h"), "dst": -10.0})
    dst.loc[dst["time"].dt.day == 12, "dst"] = -200
    assert "2025-11-12" not in events.quiet_days_from_dst(dst, sorted(allv["date"].unique()))
    s = Settings()
    s.figure_formats = ["png"]
    fig = plots.plot_storm(dev, dst, None, quiet, "test", s, None, return_fig=True)
    assert plots.text_overlaps(fig) == []


def test_events_end_to_end(tmp_path):
    days = [f"2025-11-{d:02d}" for d in range(8, 13)]
    peak = pd.Timestamp("2025-11-11 07:30")
    d = _synthetic(days, flare_peak=peak, amp=2.5)
    sta = Station(name="test", lat=23.36, lon=86.34, height=285.0)
    s = Settings()
    s.figure_formats = ["png"]
    series = []
    for day in days:
        dd = d[d["time"].dt.strftime("%Y-%m-%d") == day]
        (tmp_path / day).mkdir()
        products.write_cmn_like(dd, tmp_path / day / f"test_{day}.Cmn", sta, s)
        ser = products.daily_series(dd, day, el_min=50)
        ser["vtec_smooth"] = ser["vtec"]
        series.append(ser)
    pd.concat(series).to_csv(tmp_path / "test_vtec_all_days.csv", index=False)
    flares = tmp_path / "flares.csv"
    flares.write_text("class,start,peak,end\nX5.1,2025-11-11T07:15,2025-11-11T07:30,2025-11-11T07:45\n"
                      "X1.0,2025-11-10T18:00,2025-11-10T18:10,2025-11-10T18:20\n")
    out = events.run(tmp_path, flares_csv=flares, settings=s, log=lambda *a: None)
    summ = pd.read_csv(out / "test_flare_summary.csv")
    summ = summ.set_index("class")
    assert bool(summ.loc["X5.1", "detected"]) and "darkness" in summ.loc["X1.0", "note"]
    assert (out / "test_flare_2025-11-11_0730_X5.1.png").exists()
    assert (out / "test_flares_overview.png").exists()
    assert (out / "test_storm_dvtec.png").exists()
    tr = pd.read_csv(out / "test_flare_2025-11-11_0730_X5.1_satellites.csv", parse_dates=["time"])
    st = pd.read_csv(out / "test_flare_2025-11-11_0730_X5.1_median.csv")
    row = summ.loc["X5.1"].to_dict() | {"class": "X5.1"}
    row["peak"] = pd.Timestamp(row["peak"]); row["start"] = pd.Timestamp(row["start"]); row["end"] = pd.Timestamp(row["end"])
    assert plots.text_overlaps(plots.plot_flare(tr, st, row, "test", 30, s, None, return_fig=True)) == []
