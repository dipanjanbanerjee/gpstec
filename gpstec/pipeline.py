"""End-to-end processing: receiver file or RINEX -> calibrated TEC -> files and plots."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from . import calibrate, continuity, dcb, gim, plots, products
from .convert import gps_to_rinex
from .orbits import BroadcastOrbit, SP3Orbit
from .rinex_obs import read_obs
from .tec import add_geometry, compute_stec, join_midnight_arcs

RINEX_OBS_SUFFIXES = ("o", ".obs", ".rnx", ".crx")


def find_nav(obs_path):
    """Look for the navigation file that belongs with an observation file."""
    p = Path(obs_path)
    cands = []
    if p.suffix[-1:].lower() == "o":
        base = p.name[:-1]
        cands += [p.with_name(base + c) for c in ("n", "N", "p", "P")]
    cands += [p.with_suffix(".nav"), p.with_suffix(".22n")]
    cands += [p.with_name(p.name.replace("_MO.", "_GN.").replace("_MO.", "_MN."))]
    for c in cands:
        if c.exists() and c != p:
            return c
    return None


def prepare_file(path, station, settings, out_dir, nav=None, sp3=None, orbit=None, log=print):
    """Read, add geometry and compute levelled slant TEC (no biases yet)."""
    path = Path(path)
    out_dir = Path(out_dir)
    if path.suffix.upper() == ".GPS":
        log(f"Converting {path.name} with RTKLIB convbin ...")
        obs_path, nav_path = gps_to_rinex(path, out_dir / "rinex", station, settings)
        nav = nav or nav_path
    else:
        obs_path = path
        nav = nav or find_nav(path)

    log(f"Reading {obs_path.name} ...")
    hdr, obs = read_obs(obs_path)
    log(f"  {len(obs):,} observations, {obs['prn'].nunique()} satellites, "
        f"{obs['time'].min()} to {obs['time'].max()}")
    hx = hdr.approx_xyz
    if np.linalg.norm(hx) < 6.0e6:
        log(f"  note: header position {hx} is not a valid ECEF position; using the config station position")

    if orbit is None:
        if sp3:
            orbit = SP3Orbit(sp3)
        elif nav:
            orbit = BroadcastOrbit(nav)
        else:
            raise FileNotFoundError(f"No navigation file found for {obs_path}. Pass it with --nav.")
    log("Computing satellite geometry ...")
    g = add_geometry(obs, orbit, station, settings.shell_height_km)
    log("Computing slant TEC (code + levelled phase) ...")
    d = compute_stec(g, settings.pair, max_gap_s=settings.max_gap_s, slip_m=settings.slip_threshold_m,
                     min_arc_s=settings.min_arc_s, level_el_deg=settings.level_elevation,
                     mask_el_deg=settings.mask_elevation)
    code_used = d["code"].iloc[0]
    log(f"  {d['arc'].nunique()} satellite arcs; first-frequency code: {code_used}")
    info = {"file": str(path), "obs_file": str(obs_path), "nav_file": str(nav) if nav else None,
            "code_used": code_used, "station": station.__dict__.copy()}
    return d, info


def satellite_biases_from_files(d, settings, log=print):
    """(sat_bias dict, source) from bias files / downloads, or (None, None)."""
    if settings.sat_dcb not in ("auto", "file"):
        return None, None
    code_used = d["code"].iloc[0]
    files = [Path(f) for f in settings.dcb_files]
    if not files:
        files = dcb.find_or_fetch(d["time"].min(), Path(settings.dcb_dir), log=log,
                                  offline=settings.sat_dcb == "file")
    if files:
        sat_bias = dcb.satellite_bias_tecu(files, code_used)
        if sat_bias:
            source = ", ".join(Path(f).name for f in files)
            log(f"  satellite biases from {source}")
            return sat_bias, source
    if settings.sat_dcb == "file":
        raise FileNotFoundError("sat_dcb = 'file' but no usable bias file was found.\n"
                                + dcb.manual_download_help(d["time"].min()))
    return None, None


def gim_file(d, settings, info, log=print):
    """Path of the CODE GIM for this file's main day (or None)."""
    if settings.gim == "off":
        return None
    if "gim_file" in info:
        return info["gim_file"]
    day = d["time"].dt.floor("D").value_counts().idxmax()
    f = gim.find_or_fetch(np.datetime64(day.date()), settings.gim_dir, log=log)
    info["gim_file"] = str(f) if f else None
    if f is None:
        log("  no global ionosphere map (CODG...I) found; the GIM reference line is skipped")
    return info["gim_file"]


def receiver_bias_from_gim(sub, path, settings, info):
    """B_rx that makes the night-time VTEC of high satellites match the GIM (median)."""
    s = sub[sub["el"] >= settings.output_elevation].iloc[::30]
    st = info["station"]
    g = gim.station_series(path, st["lat"], st["lon"], s["time"].to_numpy())
    b = g / s["mf"].to_numpy() - s["stec_sat"].to_numpy()
    b = b[np.isfinite(b)]
    if len(b) < 20:
        raise ValueError("not enough night data to compare with the GIM")
    return float(np.median(b))


def estimate_receiver_bias(d, info, settings, b_rx_estimated=None, log=print):
    """Receiver bias for one file (satellite biases already applied). Returns (b_rx, how)."""
    rx_mode = settings.receiver_bias
    st = info["station"]
    sub, window = _night_subset(d, settings, info)
    diag = {}
    # least squares at several elevation cut-offs: the main value and its sensitivity
    for e in (15.0, settings.lsq_elevation, 30.0):
        try:
            diag[f"night_lsq_el{e:g}"] = calibrate.receiver_bias_lsq_fixed_sats(sub, st["lat"], st["lon"], el_min=e)
        except ValueError:
            pass
    diag["night_lsq"] = diag.get(f"night_lsq_el{settings.lsq_elevation:g}", np.nan)
    try:
        diag["night_minstd"], diag["minstd_spread_tecu"] = calibrate.receiver_bias_minstd(
            sub, el_min=settings.calib_elevation)
    except ValueError:
        pass
    gpath = gim_file(d, settings, info, log)
    if gpath:
        try:
            diag["night_gim_match"] = receiver_bias_from_gim(sub, gpath, settings, info)
        except ValueError:
            pass
    if isinstance(rx_mode, (int, float)):
        b_rx, how = float(rx_mode), "fixed in config"
    elif rx_mode == "gim":
        if "night_gim_match" not in diag:
            raise RuntimeError("receiver_bias = 'gim' but no usable CODE GIM file was found "
                               f"(put CODG*.I files in {settings.gim_dir}/)")
        b_rx, how = diag["night_gim_match"], f"matched to CODE GIM, {window}"
    elif rx_mode == "lsq":
        if b_rx_estimated is not None:
            b_rx, how = b_rx_estimated, "least squares, joint with estimated satellite biases"
        else:
            b_rx = diag["night_lsq"]
            how = f"least squares with N/E gradients, {window}, elevation >= {settings.lsq_elevation:g} deg"
    else:
        b_rx = diag["night_minstd"]
        how = f"minimum spread, {window} (mean spread {diag['minstd_spread_tecu']:.2f} TECU)"
    diag["floor"] = calibrate.receiver_bias_floor(d, el_min=settings.calib_elevation, pct=settings.floor_percentile)
    info["receiver_bias_diagnostics"] = {k: round(float(v), 3) for k, v in diag.items() if np.isfinite(v)}
    info["receiver_bias_this_day"] = b_rx
    return b_rx, how


def finish_calibration(d, info, settings, sat_bias, source, b_rx_estimated=None, log=print,
                       b_rx_override=None):
    """Apply satellite bias, find/apply receiver bias (with the physical floor), add VTEC."""
    d = calibrate.apply_sat_bias(d, sat_bias)
    info["sat_bias_source"] = source
    info["sat_bias_tecu"] = {int(k): round(v, 3) for k, v in sat_bias.items()}
    if b_rx_override is not None:
        b_rx, how = b_rx_override
        floor = calibrate.receiver_bias_floor(d, el_min=settings.calib_elevation, pct=settings.floor_percentile)
    else:
        b_rx, how = estimate_receiver_bias(d, info, settings, b_rx_estimated, log)
        floor = info["receiver_bias_diagnostics"]["floor"]
    if settings.enforce_nonnegative and b_rx < floor:
        log(f"  receiver bias {b_rx:+.2f} TECU would make VTEC negative; raised to the "
            f"physical minimum {floor:+.2f} TECU")
        how += f"; raised from {b_rx:.2f} to the non-negative-VTEC floor"
        b_rx = floor
    log(f"  receiver bias {b_rx:+.2f} TECU ({b_rx / 2.854:+.2f} ns), {how}")
    info["receiver_bias_tecu"] = b_rx
    info["receiver_bias_method"] = how
    d = calibrate.apply_receiver_bias(d, b_rx)

    hi = d[d["el"] >= settings.output_elevation]
    info["negative_vtec_fraction"] = float((hi["vtec"] < -1).mean()) if len(hi) else None
    if info["negative_vtec_fraction"] and info["negative_vtec_fraction"] > 0.02:
        log(f"  warning: {100 * info['negative_vtec_fraction']:.1f}% of VTEC values above "
            f"{settings.output_elevation:g}° are below -1 TECU; check the biases")
    return d, info


def _night_subset(d, settings, info):
    """Rows inside the night local-time window, if there is enough of them."""
    if not settings.calib_night_lt:
        return d, "whole day"
    a, b = settings.calib_night_lt
    off = info["station"]["utc_offset_h"]
    lt = ((d["time"] - d["time"].dt.floor("D")).dt.total_seconds() / 3600.0 + off) % 24
    m = (lt >= a) | (lt < b) if a > b else (lt >= a) & (lt < b)
    sub = d[m & (d["el"] >= settings.calib_elevation)]
    if sub["time"].nunique() < 1800:          # less than ~30 min of usable night data
        return d, "whole day (too little night data)"
    return d[m], f"night {a:g}-{b:g} LT"


def process_file(path, station, settings, out_dir, nav=None, sp3=None, orbit=None, log=print):
    """One file on its own: prepare + calibrate. Returns (table, info)."""
    d, info = prepare_file(path, station, settings, out_dir, nav, sp3, orbit, log)
    sat_bias, source = satellite_biases_from_files(d, settings, log)
    b_rx = None
    if sat_bias is None:
        _warn_estimating(d, log)
        sat_bias, b_rx, rms = calibrate.biases_lsq(d, el_min=settings.calib_elevation)
        source = "estimated from this day's data (least squares)"
        info["lsq_fit_rms_tecu"] = rms
    return finish_calibration(d, info, settings, sat_bias, source, b_rx, log)


def _warn_estimating(d, log):
    log("  WARNING: no satellite bias file could be found or downloaded.")
    log("  Estimating satellite biases from the station's own data instead (accuracy: a few TECU).")
    log("  " + dcb.manual_download_help(d["time"].min()).replace("\n", "\n  "))


def days_in(d, min_hours=1.0):
    t = d["time"]
    days = t.dt.floor("D").value_counts()
    # keep days that have at least min_hours of data
    keep = []
    for day in sorted(days.index):
        sub = t[(t >= day) & (t < day + pd.Timedelta(days=1))]
        if (sub.max() - sub.min()).total_seconds() >= min_hours * 3600:
            keep.append(np.datetime64(day.date()))
    return keep


def run(inputs, station, settings, out_dir, nav=None, sp3=None, log=print, write_cmn=True, orbit=None):
    """Process many files. Days without a bias file get satellite biases estimated
    jointly from ALL such days (biases are stable over weeks; more days = better)."""
    out_dir = Path(out_dir)
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    prepared, need_est, samples = [], [], []
    for i, path in enumerate(inputs):
        d, info = prepare_file(path, station, settings, out_dir, nav=nav, sp3=sp3, orbit=orbit, log=log)
        sat_bias, source = satellite_biases_from_files(d, settings, log)
        tmp = work / f"stec_{i}.pkl"
        d.to_pickle(tmp)
        info["_start"], info["_end"] = d["time"].min(), d["time"].max()
        prepared.append((tmp, info, sat_bias, source))
        del d
    prepared.sort(key=lambda p: p[1]["_start"])
    _join_across_midnight(prepared, settings, log)
    for i, (tmp, info, sat_bias, source) in enumerate(prepared):
        if sat_bias is None:
            need_est.append(i)
            samples.append(calibrate.sample_for_fit(pd.read_pickle(tmp), settings.calib_elevation))

    est_bias, est_rx = None, None
    if need_est:
        _warn_estimating(pd.read_pickle(prepared[need_est[0]][0]), log)
        joint = pd.concat(samples, ignore_index=True)
        n_days = len(need_est)
        est_bias, est_rx, rms = calibrate.biases_lsq(joint, el_min=settings.calib_elevation, presampled=True)
        log(f"  estimated satellite biases from {n_days} file(s), fit rms {rms:.2f} TECU")

    # ── receiver bias per file, then (optionally) one value for the whole run ──
    resolved = []
    for i, (tmp, info, sat_bias, source) in enumerate(prepared):
        b_est = None
        if sat_bias is None:
            sat_bias, b_est = est_bias, est_rx
            source = f"estimated from the data of {len(need_est)} file(s) (least squares)"
        d = calibrate.apply_sat_bias(pd.read_pickle(tmp), sat_bias)
        b_rx, how = estimate_receiver_bias(d, info, settings, b_est, log)
        log(f"  {Path(info['file']).name}: receiver bias estimate {b_rx:+.2f} TECU ({how})")
        resolved.append([tmp, info, sat_bias, source, b_rx, how])
        del d
    est = np.array([r[4] for r in resolved], float)
    if settings.receiver_bias_multiday == "median" and len(resolved) >= 3 and \
            not isinstance(settings.receiver_bias, (int, float)):
        med = float(np.nanmedian(est))
        log(f"  using the median receiver bias of {len(est)} files: {med:+.2f} TECU "
            f"(per-file values {np.round(est, 1).tolist()})")
        for r in resolved:
            r[4], r[5] = med, f"median of {len(est)} files ({r[5]})"

    # calibrate every file first, so each day's smooth curve can look a little past midnight
    for r in resolved:
        tmp, info, sat_bias, source, b_rx, how = r
        d, info = finish_calibration(pd.read_pickle(tmp), info, settings, sat_bias, source, log=log,
                                     b_rx_override=(b_rx, how))
        d.to_pickle(tmp)
        r[1] = info
        del d

    all_series, summary, bounds = [], [], []
    for k, (tmp, info, sat_bias, source, b_rx, how) in enumerate(resolved):
        d = pd.read_pickle(tmp)
        pads = []
        for j in (k - 1, k + 1):
            if 0 <= j < len(resolved):
                nb = resolved[j][1]
                if abs((nb["_start"] - info["_end"]).total_seconds()) <= 3600 or \
                        abs((info["_start"] - nb["_end"]).total_seconds()) <= 3600:
                    n = pd.read_pickle(resolved[j][0])
                    w = pd.Timedelta(hours=3)
                    pads.append(n[(n["time"] >= info["_start"] - w) & (n["time"] <= info["_end"] + w)])
                    del n
        info = {k_: v for k_, v in info.items() if not k_.startswith("_")} | {"_pad": pads}
        n0 = len(summary)
        series = _write_days(d, info, station, settings, out_dir, write_cmn, summary, log)
        all_series += series
        if series:
            bounds.append({"edges": continuity.edges(d), "b_rx": info["receiver_bias_tecu"],
                           "first": (str(summary[n0]["date"]), series[0]),
                           "last": (str(summary[-1]["date"]), series[-1]),
                           "unc": summary[-1].get("calibration_uncertainty_tecu"), "n0": n0})
        del d
    for r in resolved:
        r[0].unlink()
    _continuity_report(bounds, summary, all_series, station, settings, out_dir, log)
    try:
        work.rmdir()
    except OSError:
        pass

    if all_series:
        allv = pd.concat(all_series, ignore_index=True)
        allv.to_csv(out_dir / f"{station.name}_vtec_all_days.csv", index=False, float_format="%.3f")
        if allv["date"].nunique() > 1:
            plots.plot_overlay(allv, station, settings, out_dir / f"{station.name}_vtec_overlay.png")
            plots.plot_heatmap(allv, station, settings, out_dir / f"{station.name}_vtec_heatmap.png")
        pd.DataFrame([{k: v for k, v in s.items() if k in
                       ("date", "file", "code_used", "sat_bias_source", "receiver_bias_tecu",
                        "receiver_bias_method", "peak_vtec", "coverage_fraction", "negative_vtec_fraction",
                        "night_minus_gim_tecu", "midnight_jump_to_next_day_tecu", "midnight_check")}
                      for s in summary]).to_csv(out_dir / "calibration_summary.csv", index=False)
    return summary


def _join_across_midnight(prepared, settings, log):
    """Level satellite passes that cross midnight as one pass (see tec.join_midnight_arcs)."""
    for (ta, ia, *_), (tb, ib, *_) in zip(prepared[:-1], prepared[1:]):
        gap = (ib["_start"] - ia["_end"]).total_seconds()
        if not (0 < gap <= settings.max_gap_s):
            continue
        a, b = pd.read_pickle(ta), pd.read_pickle(tb)
        done = join_midnight_arcs(a, b, settings.pair, settings.max_gap_s, settings.slip_threshold_m,
                                  settings.level_elevation)
        if done:
            a.to_pickle(ta)
            b.to_pickle(tb)
            moved = ", ".join(f"G{p:02d} {oa - off:+.1f}/{ob - off:+.1f}" for p, oa, ob, off in done)
            log(f"  {Path(ia['file']).name} -> {Path(ib['file']).name}: {len(done)} satellite passes cross "
                f"midnight and were levelled as one (change before/after midnight, TECU: {moved})")
            for info, side in ((ia, "end"), (ib, "start")):
                info.setdefault("passes_joined_across_midnight", {})[side] = [
                    {"prn": p, "offset_change_tecu": round((off - (oa if side == "end" else ob)), 2)}
                    for p, oa, ob, off in done]
        del a, b


def _continuity_report(bounds, summary, all_series, station, settings, out_dir, log):
    """Check that each day ends where the next one starts (see continuity.py)."""
    bounds = sorted(bounds, key=lambda b: b["edges"]["start"])
    rows, per_all = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        prev = {"edges": a["edges"], "b_rx": a["b_rx"], "date": a["last"][0], "series": a["last"][1], "unc": a["unc"]}
        nxt = {"edges": b["edges"], "b_rx": b["b_rx"], "date": b["first"][0], "series": b["first"][1], "unc": b["unc"]}
        got = continuity.compare(prev, nxt, settings.output_elevation)
        if got is None:
            log(f"  midnight check {prev['date']} -> {nxt['date']}: files are not consecutive, skipped")
            continue
        res, per = got
        rows.append(res)
        if len(per):
            per.insert(0, "boundary_ut", res["boundary_ut"])
            per_all.append(per)
        j = res.get("jump_median_el30")
        if j is not None:
            log(f"  midnight check {prev['date']} -> {nxt['date']}: VTEC jump {j:+.2f} TECU "
                f"({res['n_sats_el30']} satellites, of which receiver bias {res['jump_from_rx_bias']:+.2f}, "
                f"arc levelling {res['jump_from_levelling']:+.2f}); smooth curve jump "
                f"{res.get('smooth_jump', float('nan')):+.2f}  [{res['status']}]")
        for s in summary:
            if str(s["date"]) == prev["date"]:
                s["midnight_jump_to_next_day_tecu"] = j
                s["midnight_check"] = res["status"]
                s["midnight_continuity"] = res
        day_json = out_dir / prev["date"] / f"{station.name}_{prev['date']}_calibration.json"
        if day_json.exists():
            info = json.loads(day_json.read_text())
            info["midnight_continuity"] = res
            day_json.write_text(json.dumps(info, indent=2, default=str))
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "midnight_continuity.csv", index=False)
        if per_all:
            pd.concat(per_all).to_csv(out_dir / "midnight_continuity_per_satellite.csv", index=False)
        try:
            plots.plot_continuity(pd.concat(all_series, ignore_index=True), rows, station, settings, out_dir / f"{station.name}_midnight_continuity.png")
        except Exception as e:                              # a plot must never stop the run
            log(f"  continuity plot skipped ({type(e).__name__}: {e})")


def _write_days(d, info, station, settings, out_dir, write_cmn, summary, log):
    out = []
    for day in days_in(d):
        tag = f"{station.name}_{day}"
        day_dir = out_dir / str(day)
        day_dir.mkdir(exist_ok=True)
        s = products.daily_series(d, day, el_min=settings.output_elevation, bin_min=settings.bin_minutes,
                                  nsig=settings.clip_sigma, min_sats=settings.min_sats_per_bin)
        dpad = pd.concat([d] + info.get("_pad", []), ignore_index=True) if info.get("_pad") else d
        sm = products.smooth_station_vtec(dpad, day, station.lat, station.lon, el_min=settings.output_elevation,
                                          knot_min=settings.smooth_knot_min, grid_min=settings.bin_minutes)
        for col in ("vtec_smooth", "grad_north", "grad_east"):
            s[col] = sm[col].to_numpy()
        # calibration uncertainty: spread of the independent receiver-bias estimates
        diag = info.get("receiver_bias_diagnostics", {})
        ests = [v for k, v in diag.items() if k.startswith("night_lsq_el") or k == "night_gim_match"]
        ests = [e for e in ests if e is not None and np.isfinite(e)]
        unc = max(0.5, float(np.std(ests))) if len(ests) > 1 else None
        s["vtec_uncertainty"] = unc * 0.9 if unc else np.nan        # mf ~0.9 above 50 deg
        if info.get("gim_file"):
            t = day + (s["ut_hour"].to_numpy() * 3.6e12).astype("timedelta64[ns]")
            s["vtec_gim"] = gim.station_series(info["gim_file"], station.lat, station.lon, t)
        s.to_csv(day_dir / f"{tag}_vtec_{settings.bin_minutes:g}min.csv", index=False, float_format="%.3f")
        products.write_std_like(s, day_dir / f"{tag}.Std", station)
        dd = d[(d["time"] >= day) & (d["time"] < day + np.timedelta64(1, "D"))]
        if write_cmn:
            products.write_cmn_like(dd, day_dir / f"{tag}.Cmn", station, settings, source=info["file"])
        plots.plot_day(d, s, station, settings, day_dir / f"{tag}_vtec.png")
        plots.plot_ipp_map(d, station, settings, day_dir / f"{tag}_pierce_points.png", day)
        out.append(s)
        extra = {}
        if "vtec_gim" in s:
            lt = (s["ut_hour"] + station.utc_offset_h) % 24
            a, b = settings.calib_night_lt or [22.0, 6.0]
            night = (lt >= a) | (lt < b)
            dif = (s["vtec_smooth"] - s["vtec_gim"])[night]
            extra["night_minus_gim_tecu"] = float(np.nanmedian(dif)) if dif.notna().any() else None
            log(f"  {day}: station VTEC minus CODE GIM at night = {extra['night_minus_gim_tecu']:+.2f} TECU")
        if s["vtec_uncertainty"].notna().any():
            extra["calibration_uncertainty_tecu"] = float(s["vtec_uncertainty"].iloc[0])
        day_info = dict({k: v for k, v in info.items() if not k.startswith("_")}, **extra, date=str(day),
                        peak_vtec=float(np.nanmax(s["vtec_smooth"])) if s["vtec_smooth"].notna().any() else None,
                        coverage_fraction=float(s["vtec"].notna().mean()))
        with open(day_dir / f"{tag}_calibration.json", "w") as fh:
            json.dump(day_info, fh, indent=2, default=str)
        summary.append(day_info)
        log(f"  wrote {day_dir}")
    return out
