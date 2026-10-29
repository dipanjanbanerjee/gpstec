import numpy as np
import pandas as pd

from gpstec import continuity
from gpstec.tec import join_midnight_arcs, PAIRS


def _arc(t0, n, l4_0, rate, off, prn=5, el0=50.0):
    t = pd.Timestamp(t0) + pd.to_timedelta(np.arange(n), unit="s")
    l4 = l4_0 + rate * np.arange(n)
    K = PAIRS["L1L2"][6]
    true_tec_m = 2.0 + 0.0 * l4                       # code = phase + constant (no noise)
    p4 = l4 + true_tec_m
    return pd.DataFrame({"time": t, "prn": prn, "el": el0 + 0.001 * np.arange(n), "p4": p4, "l4": l4,
                         "arc": 1, "level_offset": off, "l4_lev": l4 + off, "stec_raw": K * (l4 + off),
                         "arc_rms": 0.0})


def test_join_midnight_arcs_gives_one_offset():
    a = _arc("2022-12-25 23:00:00", 3601, 0.0, 1e-4, off=2.3)      # ends 2022-12-26 00:00:00
    b = _arc("2022-12-26 00:00:01", 1800, 0.3601, 1e-4, off=1.6)   # phase continues
    done = join_midnight_arcs(a, b)
    assert len(done) == 1
    assert np.isclose(a["level_offset"].iloc[-1], b["level_offset"].iloc[0])
    assert np.isclose(a["level_offset"].iloc[0], 2.0, atol=1e-6)    # the true code-phase offset


def test_join_skips_cycle_slip():
    a = _arc("2022-12-25 23:00:00", 3601, 0.0, 1e-4, off=2.3)
    b = _arc("2022-12-26 00:00:01", 1800, 5.0, 1e-4, off=1.6)      # 5 m jump = slip
    assert join_midnight_arcs(a, b) == []


def test_continuity_reports_bias_step():
    def edge(start, end, offset):
        t = pd.date_range(start, end, freq="1s")
        rows = []
        for prn in (1, 2, 3):
            rows.append(pd.DataFrame({"time": t, "prn": prn, "el": 60.0, "vtec": 4.0 + offset,
                                      "stec": (4.0 + offset) / 0.9, "arc": prn}))
        r = pd.concat(rows, ignore_index=True)
        return {"start": t[0], "end": t[-1], "rows": r}
    ea = edge("2022-12-25 23:50:00", "2022-12-26 00:00:00", 0.0)
    eb = edge("2022-12-26 00:00:01", "2022-12-26 00:10:00", 0.9)
    prev = {"edges": ea, "b_rx": 32.0, "date": "2022-12-25", "series": None, "unc": 1.0}
    nxt = {"edges": eb, "b_rx": 33.0, "date": "2022-12-26", "series": None, "unc": 1.0}
    res, per = continuity.compare(prev, nxt, 50.0)
    assert np.isclose(res["jump_median_el30"], 0.9, atol=0.01)
    assert np.isclose(res["jump_from_rx_bias"], 0.9, atol=0.01)
    assert abs(res["jump_from_levelling"]) < 0.02
    assert res["status"] == "OK"
