"""Publication-quality VTEC figures.

Every figure is laid out with matplotlib's constrained layout, legends sit outside
the data area, and each figure is written in all formats listed in
``settings.figure_formats`` (vector PDF/SVG/EPS and 300 dpi PNG/TIFF by default).
``text_overlaps`` lets the tests prove that no two pieces of text touch.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches
import matplotlib.text
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#ffffff"
BLUE = "#2a78d6"
BLUE_LIGHT = "#b7d3f6"
ORANGE = "#eb6834"
RED = "#e34948"
SAT_GRAY = "#bdbcb6"
# categorical order (validated for colour-vision deficiency on white)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,          # editable, embedded TrueType text in PDF
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.constrained_layout.h_pad": 0.04,
    "figure.constrained_layout.w_pad": 0.04,
}

HALO = [pe.withStroke(linewidth=2.5, foreground=SURFACE)]


def _fig(settings, height_ratio, **kw):
    w = getattr(settings, "figure_width_in", 7.2)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(w, w * height_ratio), facecolor=SURFACE, layout="constrained")
    return fig


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, which="both")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _hour_axis(ax, utc_offset_h=None, label_top=True):
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticks(range(0, 25, 1), minor=True)
    if utc_offset_h is not None:
        top = ax.secondary_xaxis("top", functions=(lambda x: x + utc_offset_h, lambda x: x - utc_offset_h))
        ticks = [h for h in range(0, 31, 3) if 0 <= h - utc_offset_h <= 24]
        top.set_xticks(ticks)
        top.set_xticklabels([f"{h % 24:d}" for h in ticks])
        top.tick_params(colors=INK2)
        top.spines["top"].set_color(INK2)
        if label_top:
            top.set_xlabel(f"Local time (UT{utc_offset_h:+g} h)", color=INK2)
        return top
    return None


def _title(ax, settings, text):
    if getattr(settings, "figure_titles", True):
        ax.set_title(text, loc="left", color=INK)


def _legend(fig, handles, ncol, loc="outside lower center"):
    leg = fig.legend(handles=handles, loc=loc, ncol=ncol, frameon=False, handlelength=2.2,
                     columnspacing=1.6, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def save(fig, path, settings):
    """Write the figure in every configured format; returns the written paths."""
    path = Path(path)
    out = []
    with plt.rc_context(RC):
        for fmt in getattr(settings, "figure_formats", ["png"]) or ["png"]:
            p = path.with_suffix("." + fmt.lower().lstrip("."))
            kw = {"dpi": getattr(settings, "figure_dpi", 300)} if fmt.lower() in ("png", "tiff", "tif", "jpg") else {}
            if fmt.lower() in ("tiff", "tif"):
                kw["pil_kwargs"] = {"compression": "tiff_lzw"}
            fig.savefig(p, facecolor=SURFACE, **kw)
            out.append(p)
    plt.close(fig)
    return out


def _drawn_texts(fig):
    """Every piece of text that is actually drawn (ticks outside the view are skipped)."""
    out = list(fig.texts)
    for leg in fig.legends:
        out += leg.get_texts()
    axes = []
    for ax in fig.axes:
        axes += [ax] + list(getattr(ax, "child_axes", []))
    for ax in axes:
        out += [ax.title, ax._left_title, ax._right_title] + list(ax.texts)
        if ax.get_legend():
            out += ax.get_legend().get_texts()
        for axis in (ax.xaxis, ax.yaxis):
            if not axis.get_visible():
                continue
            out.append(axis.label)
            for tick in axis._update_ticks():
                out += [lab for lab in (tick.label1, tick.label2) if lab.get_visible()]
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def text_overlaps(fig, pad_px=0.5):
    """Pairs of drawn text items whose boxes overlap, or text outside the figure
    (an empty list means a clean layout)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for t in _drawn_texts(fig):
        bb = t.get_window_extent(r)
        if bb.width > 0 and bb.height > 0:
            items.append((t, bb.padded(-pad_px)))
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][1].overlaps(items[j][1]):
                bad.append((items[i][0].get_text(), items[j][0].get_text()))
    W, H = fig.bbox.width, fig.bbox.height
    for t, bb in items:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            bad.append((t.get_text(), "<outside figure>"))
    return bad


# ───────────────────────────── daily figure ─────────────────────────────
def _interval_s(times):
    """Sampling interval of the data in seconds (median step between epochs)."""
    t = np.unique(np.asarray(times, dtype="datetime64[ns]"))
    return float(np.median(np.diff(t)) / np.timedelta64(1, "s")) if len(t) > 1 else np.nan


def _fmt_s(sec):
    if not np.isfinite(sec):
        return "?"
    if sec >= 3600:
        return f"{sec / 3600:g} h"
    return f"{sec / 60:g} min" if sec >= 60 else f"{sec:g} s"


def _gim_points(series, gim_hourly):
    """(UT hour, VTEC) of the CODE maps themselves - one point per published map."""
    if gim_hourly is not None and len(gim_hourly):
        return gim_hourly["ut_hour"].to_numpy(float), gim_hourly["vtec_gim"].to_numpy(float), gim_hourly.attrs
    if "vtec_gim" in series and series["vtec_gim"].notna().any():
        g = series.dropna(subset=["vtec_gim"])
        return g["ut_hour"].to_numpy(float), g["vtec_gim"].to_numpy(float), {}
    return None, None, {}


def _gim_label(attrs):
    step = attrs.get("interval_s")
    grid = f", {attrs['dlat']:g}° × {attrs['dlon']:g}° grid" if attrs.get("dlat") else ""
    every = f"one map every {_fmt_s(step)}" if step else "hourly maps"
    return f"CODE global map ({every}{grid})"


def _main_col(series):
    """The station curve to draw: the gradient-corrected fit when present, else the bin means."""
    if "vtec_smooth" in series and series["vtec_smooth"].notna().any():
        return "vtec_smooth"
    return "vtec"


def plot_day(d, series, station, settings, path, title_extra="", return_fig=False, gim_hourly=None):
    """One day: every satellite's VTEC at the receiver sampling, the 1-min mean of the
    satellites, the gradient-corrected station VTEC, and the CODE map at its map epochs."""
    day = np.datetime64(series["date"].iloc[0], "D")
    el = settings.output_elevation
    s = d[(d["el"] >= el) & (d["time"] >= day) & (d["time"] < day + np.timedelta64(1, "D"))]
    dt_rx = _interval_s(s["time"]) if len(s) else np.nan
    binm = settings.bin_minutes
    fig = _fig(settings, 0.70)
    with plt.rc_context(RC):
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
        ax = fig.add_subplot(gs[0])
        axn = fig.add_subplot(gs[1], sharex=ax)
        _style(ax)
        _style(axn)

        gap_h = max(2 / 60, 3 * dt_rx / 3600) if np.isfinite(dt_rx) else 2 / 60
        for prn, g in s.groupby("prn"):
            h = (g["time"] - day).dt.total_seconds().to_numpy() / 3600
            v = g["vtec"].to_numpy()
            brk = np.where(np.diff(h) > gap_h)[0] + 1          # never draw across a data gap
            for hh, vv in zip(np.split(h, brk), np.split(v, brk)):
                ax.plot(hh, vv, color=SAT_GRAY, lw=0.6, zorder=1, solid_capstyle="round", rasterized=True)

        x = series["ut_hour"].to_numpy() + binm / 120.0           # centre of each bin
        raw = series["vtec"].to_numpy()
        col = _main_col(series)
        m = series[col].to_numpy()
        handles = [Line2D([], [], color=SAT_GRAY, lw=1.2, label=f"Each satellite ≥ {el:g}° (every {_fmt_s(dt_rx)})")]
        if col == "vtec_smooth":
            ax.plot(x, raw, color=INK2, lw=0.6, zorder=2)
            handles.append(Line2D([], [], color=INK2, lw=0.8, label=f"Mean of satellites ({binm:g} min)"))
        if "vtec_uncertainty" in series and series["vtec_uncertainty"].notna().any():
            u = series["vtec_uncertainty"].to_numpy()
            ax.fill_between(x, m - u, m + u, color=BLUE_LIGHT, lw=0, zorder=2)
            handles.append(matplotlib.patches.Patch(color=BLUE_LIGHT,
                                                    label=f"Calibration uncertainty (±{np.nanmax(u):.1f} TECU)"))
        gx, gy, gattr = _gim_points(series, gim_hourly)
        if gx is not None:
            ax.plot(gx, gy, color=ORANGE, lw=1.0, ls=(0, (4, 2)), marker="o", ms=2.8, zorder=3)
            handles.append(Line2D([], [], color=ORANGE, lw=1.0, ls=(0, (4, 2)), marker="o", ms=3,
                                  label=_gim_label(gattr)))
        ax.plot(x, m, color=BLUE, lw=2.0, zorder=4)
        if col == "vtec_smooth":
            handles.append(Line2D([], [], color=BLUE, lw=2.0,
                                  label=f"Station VTEC (gradient-corrected fit, {settings.smooth_knot_min:g}-min knots)"))
        else:
            handles.append(Line2D([], [], color=BLUE, lw=2.0, label=f"Station VTEC (mean of satellites, {binm:g} min)"))

        vals = np.r_[m[np.isfinite(m)], s["vtec"].to_numpy() if len(s) else [],
                     gy[np.isfinite(gy)] if gx is not None else [], 10.0]
        ymax = float(np.nanmax(vals))
        ymin = min(0.0, float(np.nanmin(np.r_[s["vtec"].to_numpy() if len(s) else [0.0], 0.0])))
        ax.set_ylim(ymin, ymax * 1.16)
        ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        ax.set_ylabel("VTEC (TECU)", color=INK)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        if np.isfinite(m).any():
            i = int(np.nanargmax(m))
            ax.plot([x[i]], [m[i]], "o", ms=3.5, color=BLUE, mec=SURFACE, mew=1, zorder=7)
            ax.annotate(f"{m[i]:.1f} TECU, {x[i]:.1f} UT", (x[i], m[i]), xytext=(6, 5),
                        textcoords="offset points", ha="left" if x[i] < 18 else "right",
                        color=INK, fontsize=7.5, path_effects=HALO, zorder=8)
        ax.tick_params(labelbottom=False)
        _hour_axis(ax, station.utc_offset_h)
        _title(ax, settings, f"{station.name.upper()}   {day}   Vertical TEC, elevation ≥ {el:g}°{title_extra}")

        axn.bar(x, series["n_sats"], width=binm / 60.0, color=SAT_GRAY, lw=0)
        axn.set_ylabel(f"Satellites\nper {binm:g} min", color=INK)
        axn.set_xlabel("Universal time (h)", color=INK)
        axn.set_ylim(0, max(4, float(np.nanmax(series["n_sats"])) + 1))
        axn.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        axn.set_xlim(0, 24)
        _legend(fig, handles, ncol=2)
        fig.align_ylabels([ax, axn])
    if return_fig:
        return fig
    return save(fig, path, settings)


def plot_stec_day(d, day, station, settings, path, return_fig=False):
    """One day of slant TEC: every satellite at the receiver sampling, with its elevation."""
    day = np.datetime64(day, "D")
    el = settings.output_elevation
    s = d[(d["time"] >= day) & (d["time"] < day + np.timedelta64(1, "D")) & d["stec"].notna()]
    dt_rx = _interval_s(s["time"]) if len(s) else np.nan
    gap_h = max(2 / 60, 3 * dt_rx / 3600) if np.isfinite(dt_rx) else 2 / 60
    fig = _fig(settings, 0.72)
    with plt.rc_context(RC):
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.3])
        ax = fig.add_subplot(gs[0])
        bx = fig.add_subplot(gs[1], sharex=ax)
        _style(ax)
        _style(bx)
        for k, (prn, g) in enumerate(s.groupby("prn")):
            c = SERIES[k % len(SERIES)]
            h = (g["time"] - day).dt.total_seconds().to_numpy() / 3600
            brk = np.where(np.diff(h) > gap_h)[0] + 1
            y_all, e_all = g["stec"].to_numpy(), g["el"].to_numpy()
            for hh, yy, ee in zip(np.split(h, brk), np.split(y_all, brk), np.split(e_all, brk)):
                hi = ee >= el
                ax.plot(hh, yy, color=GRID, lw=0.8, zorder=1, rasterized=True)
                ax.plot(hh, np.where(hi, yy, np.nan), color=c, lw=0.9, zorder=2, rasterized=True)
                bx.plot(hh, np.where(hi, ee, np.nan), color=c, lw=0.9, rasterized=True)
                bx.plot(hh, ee, color=GRID, lw=0.8, zorder=0, rasterized=True)
        ax.set_ylabel("Slant TEC (TECU)", color=INK)
        ax.set_ylim(bottom=min(0.0, float(np.nanmin(s["stec"])) if len(s) else 0.0))
        ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        ax.tick_params(labelbottom=False)
        _hour_axis(ax, station.utc_offset_h)
        _title(ax, settings, f"{station.name.upper()}   {day}   Slant TEC of each satellite (calibrated)")
        bx.axhline(el, color=INK2, lw=0.7, ls=(0, (3, 2)))
        bx.set_ylim(0, 90)
        bx.set_yticks([0, 30, 60, 90])
        bx.set_ylabel("Elevation (°)", color=INK)
        bx.set_xlabel("Universal time (h)", color=INK)
        bx.set_xlim(0, 24)
        handles = [Line2D([], [], color=SERIES[0], lw=1.2, label=f"Satellite at ≥ {el:g}° (colour = satellite)"),
                   Line2D([], [], color=GRID, lw=1.6, label=f"Same satellite below {el:g}° (not used for VTEC)"),
                   Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 2)), label=f"{el:g}° cut-off"),
                   Line2D([], [], color="none", label=f"Every {_fmt_s(dt_rx)} measurement")]
        _legend(fig, handles, ncol=2)
        fig.align_ylabels([ax, bx])
    if return_fig:
        return fig
    return save(fig, path, settings)


# ───────────────────────────── multi-day figures ─────────────────────────────
def plot_overlay(all_series, station, settings, path, return_fig=False):
    """Station VTEC of several days on one 0-24 UT axis."""
    days = sorted(all_series["date"].astype(str).unique())
    fig = _fig(settings, 0.52)
    with plt.rc_context(RC):
        ax = fig.add_subplot()
        _style(ax)
        many = len(days) > len(SERIES)
        handles = []
        for i, day in enumerate(days):
            s = all_series[all_series["date"].astype(str) == day]
            color = SAT_GRAY if many else SERIES[i]
            ax.plot(s["ut_hour"] + settings.bin_minutes / 120.0, s[_main_col(all_series)], color=color,
                    lw=1.0 if many else 1.5)
            if not many:
                handles.append(Line2D([], [], color=color, lw=1.5, label=day))
        if many:
            piv = all_series.pivot_table(index="ut_hour", columns="date", values=_main_col(all_series))
            ax.plot(piv.index, piv.mean(axis=1), color=BLUE, lw=2)
            handles = [Line2D([], [], color=SAT_GRAY, lw=1, label="Individual days"),
                       Line2D([], [], color=BLUE, lw=2, label=f"Mean of {len(days)} days")]
        _hour_axis(ax, station.utc_offset_h)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Universal time (h)", color=INK)
        ax.set_ylabel("VTEC (TECU)", color=INK)
        _title(ax, settings, f"{station.name.upper()}   {days[0]} to {days[-1]}   Vertical TEC, elevation ≥ "
                             f"{settings.output_elevation:g}°")
        _legend(fig, handles, ncol=1, loc="outside right upper")
    if return_fig:
        return fig
    return save(fig, path, settings)


def plot_heatmap(all_series, station, settings, path, return_fig=False):
    """Day × UT map of station VTEC (gaps blank)."""
    piv = all_series.pivot_table(index="date", columns="ut_hour", values=_main_col(all_series), dropna=False)
    n = len(piv)
    fig = _fig(settings, min(1.2, 0.18 + 0.055 * n + 0.12))
    with plt.rc_context(RC):
        ax = fig.add_subplot()
        cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
        cmap = cmap.with_extremes(bad=SURFACE)
        mesh = ax.pcolormesh(np.r_[piv.columns.to_numpy(), 24.0], np.arange(n + 1),
                             np.ma.masked_invalid(piv.to_numpy()), cmap=cmap, shading="flat",
                             vmin=0, rasterized=True)
        step = max(1, int(np.ceil(n / 16)))
        ax.set_yticks(np.arange(0, n, step) + 0.5)
        ax.set_yticklabels([str(v) for v in piv.index[::step]], color=INK2)
        ax.invert_yaxis()
        _hour_axis(ax, None)
        ax.tick_params(colors=INK2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("Universal time (h)", color=INK)
        cb = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.04, aspect=max(8, 4 * min(n, 6)))
        cb.set_label("VTEC (TECU)", color=INK)
        cb.ax.tick_params(colors=INK2)
        cb.outline.set_visible(False)
        _title(ax, settings, f"{station.name.upper()}   Vertical TEC by day, elevation ≥ {settings.output_elevation:g}°")
    if return_fig:
        return fig
    return save(fig, path, settings)


def plot_ipp_map(d, station, settings, path, day, return_fig=False):
    """Pierce-point tracks of the day coloured by VTEC (sky coverage check)."""
    day = np.datetime64(day, "D")
    s = d[(d["time"] >= day) & (d["time"] < day + np.timedelta64(1, "D"))]
    w = getattr(settings, "figure_width_in", 7.2)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(min(w, 5.2), min(w, 5.2) * 0.98), facecolor=SURFACE, layout="constrained")
        ax = fig.add_subplot()
        _style(ax)
        cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE[1:])
        low = s[s["el"] < settings.output_elevation]
        high = s[s["el"] >= settings.output_elevation]
        ax.scatter(low["ipp_lon"], low["ipp_lat"], s=1.5, color=GRID, lw=0, rasterized=True)
        sc = ax.scatter(high["ipp_lon"], high["ipp_lat"], s=2.5, c=high["vtec"], cmap=cmap, vmin=0, lw=0,
                        rasterized=True)
        ax.plot(station.lon, station.lat, marker="^", ms=7, color=INK, mec=SURFACE, mew=1, ls="none")
        ax.set_xlabel("Pierce-point longitude (°E)", color=INK)
        ax.set_ylabel("Pierce-point latitude (°N)", color=INK)
        ax.set_aspect(1 / np.cos(np.radians(station.lat)), adjustable="datalim")
        cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label(f"VTEC (TECU), elevation ≥ {settings.output_elevation:g}°", color=INK)
        cb.ax.tick_params(colors=INK2)
        cb.outline.set_visible(False)
        handles = [Line2D([], [], marker="o", ls="none", color="#cfcec9", ms=5,
                          label=f"Below {settings.output_elevation:g}° (not used)"),
                   Line2D([], [], marker="^", ls="none", color=INK, ms=6, label=f"Station {station.name.upper()}")]
        _legend(fig, handles, ncol=2)
        every = _fmt_s(_interval_s(s["time"]))
        _title(ax, settings, f"{station.name.upper()}   {day}   Pierce points at {settings.shell_height_km:g} km, every {every}")
    if return_fig:
        return fig
    return save(fig, path, settings)


DAY_COLORS = [BLUE, ORANGE]


def _days_axis(ax, days, d0):
    """Continuous multi-day time axis: day labels at noon, dotted lines at midnight UT."""
    nd = len(days)
    step = max(1, int(np.ceil(nd / 10)))
    for k in range(1, nd):
        ax.axvline(24 * k, color=INK2, lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax.set_xlim(0, 24 * nd)
    ax.set_xticks([24 * k + 12 for k in range(0, nd, step)])
    ax.set_xticks([24 * k for k in range(nd + 1)], minor=True)
    ax.tick_params(axis="x", which="major", length=0)
    ax.set_xticklabels([days[k][5:] for k in range(0, nd, step)])
    ax.grid(False, axis="x")


def _jump_panel(bx, rows, days, key, tol_scale=1.0, label="Jump (TECU)"):
    nd = len(days)
    step = max(1, int(np.ceil(nd / 10)))
    xs, js, tol, col = [], [], [], []
    for r in rows:
        if r.get(key) is None or r["next_date"] not in days:
            continue
        t = r["tolerance_tecu"] * tol_scale
        xs.append(24 * days.index(r["next_date"]))
        js.append(r[key])
        tol.append(t)
        col.append(BLUE if abs(r[key]) <= t else RED)
    for x, t in zip(xs, tol):
        bx.fill_between([x - 2.5, x + 2.5], -t, t, color=BLUE_LIGHT, lw=0, zorder=1)
    bx.scatter(xs, js, s=26, c=col, edgecolors=SURFACE, linewidths=0.8, zorder=3)
    if nd <= 12:
        for x, j in zip(xs, js):
            bx.annotate(f"{j:+.1f}", (x, j), xytext=(7, 0), textcoords="offset points", va="center",
                        fontsize=7.5, color=INK, zorder=4)
    lim = max(3.0, 1.35 * max([abs(v) for v in js] + tol)) if js else 3.0
    bx.set_ylim(-lim, lim)
    bx.axhline(0, color=INK2, lw=0.6)
    bx.yaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
    bx.set_ylabel(label, color=INK)
    bx.set_xticks([24 * k for k in range(1, nd, step)])
    bx.set_xticklabels([f"{days[k][5:]} 00 UT" for k in range(1, nd, step)], fontsize=7.5)
    bx.grid(False, axis="x")
    bx.set_xlim(0, 24 * nd)


def plot_continuity(all_series, rows, station, settings, path, return_fig=False):
    """All days on one continuous time axis (alternate colours = separate days), the CODE map,
    and the VTEC jump at each midnight seen by the same satellites."""
    days = sorted(all_series["date"].astype(str).unique())
    nd = len(days)
    d0 = np.datetime64(days[0])
    col = _main_col(all_series)
    fig = _fig(settings, 0.62)
    with plt.rc_context(RC):
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.25])
        ax = fig.add_subplot(gs[0])
        bx = fig.add_subplot(gs[1])
        _style(ax)
        _style(bx)
        have_gim = "vtec_gim" in all_series and all_series["vtec_gim"].notna().any()
        for k, day in enumerate(days):
            s = all_series[all_series["date"].astype(str) == day]
            x = (np.datetime64(day) - d0).astype("timedelta64[h]").astype(float) + s["ut_hour"].to_numpy() \
                + settings.bin_minutes / 120.0
            if have_gim and s["vtec_gim"].notna().any():
                g = s["vtec_gim"].notna().to_numpy()
                ax.plot(x[g], s["vtec_gim"].to_numpy()[g], color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=2)
            ax.plot(x, s[col], color=DAY_COLORS[k % 2], lw=1.6, zorder=3)
        _days_axis(ax, days, d0)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("VTEC (TECU)", color=INK)
        _title(ax, settings, f"{station.name.upper()}   Day-to-day continuity, elevation ≥ "
                             f"{settings.output_elevation:g}°   (dotted lines = midnight UT)")
        _jump_panel(bx, rows, days, "jump_median_el30", label="Jump at\nmidnight (TECU)")
        bx.set_xlabel("Next day's first value minus previous day's last value, same satellites", color=INK)
        what = (f"gradient-corrected fit, {settings.smooth_knot_min:g}-min knots" if col == "vtec_smooth"
                else f"mean of satellites, {settings.bin_minutes:g} min")
        handles = [Line2D([], [], color=BLUE, lw=1.6, label=f"Station VTEC ({what}); colours alternate by day"),
                   ]
        if have_gim:
            handles.append(Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 2)), label="CODE global map (hourly maps)"))
        handles += [Line2D([], [], marker="o", ls="none", color=BLUE, ms=5, label="Jump within calibration uncertainty"),
                    Line2D([], [], marker="o", ls="none", color=RED, ms=5, label="Jump outside it"),
                    matplotlib.patches.Patch(color=BLUE_LIGHT, label="Calibration uncertainty")]
        _legend(fig, handles, ncol=2)
        fig.align_ylabels([ax, bx])
    if return_fig:
        return fig
    return save(fig, path, settings)


def plot_stec_continuity(stec, rows, station, settings, path, return_fig=False):
    """Slant TEC of every satellite above the cut-off over all days on one time axis,
    plus the slant-TEC jump at each midnight (same satellites)."""
    days = sorted(stec["date"].astype(str).unique())
    d0 = np.datetime64(days[0])
    el = settings.output_elevation
    fig = _fig(settings, 0.62)
    with plt.rc_context(RC):
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.25])
        ax = fig.add_subplot(gs[0])
        bx = fig.add_subplot(gs[1])
        _style(ax)
        _style(bx)
        s = stec[stec["el"] >= el]
        x_all = (s["time"].to_numpy() - d0) / np.timedelta64(1, "h")
        dt = _interval_s(s["time"]) if len(s) else np.nan
        for k, day in enumerate(days):
            m = (s["date"].astype(str) == day).to_numpy()
            for prn, g in s[m].groupby("prn"):
                x = x_all[m][(s[m]["prn"] == prn).to_numpy()]
                y = g["stec"].to_numpy()
                brk = np.where(np.diff(x) > 0.1)[0] + 1
                for xx, yy in zip(np.split(x, brk), np.split(y, brk)):
                    ax.plot(xx, yy, color=DAY_COLORS[k % 2], lw=0.8, zorder=2, rasterized=True)
        _days_axis(ax, days, d0)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Slant TEC (TECU)", color=INK)
        _title(ax, settings, f"{station.name.upper()}   Slant TEC of each satellite ≥ {el:g}°, all days   "
                             f"(dotted lines = midnight UT)")
        _jump_panel(bx, rows, days, "stec_jump_median", tol_scale=1 / 0.9, label="STEC jump at\nmidnight (TECU)")
        bx.set_xlabel("Next day's first value minus previous day's last value, same satellites", color=INK)
        handles = [Line2D([], [], color=BLUE, lw=1.2, label=f"Each satellite (every {_fmt_s(dt)}); colours alternate by day"),
                   Line2D([], [], marker="o", ls="none", color=BLUE, ms=5, label="Jump within calibration uncertainty"),
                   Line2D([], [], marker="o", ls="none", color=RED, ms=5, label="Jump outside it")]
        _legend(fig, handles, ncol=2)
        fig.align_ylabels([ax, bx])
    if return_fig:
        return fig
    return save(fig, path, settings)


def plot_stec_overlay(stec, station, settings, path, return_fig=False):
    """Slant TEC of every satellite above the cut-off, each day in its own colour on 0-24 UT."""
    days = sorted(stec["date"].astype(str).unique())
    el = settings.output_elevation
    many = len(days) > len(SERIES)
    fig = _fig(settings, 0.52)
    with plt.rc_context(RC):
        ax = fig.add_subplot()
        _style(ax)
        s = stec[stec["el"] >= el]
        dt = _interval_s(s["time"]) if len(s) else np.nan
        handles = []
        for i, day in enumerate(days):
            c = SAT_GRAY if many else SERIES[i]
            g0 = s[s["date"].astype(str) == day]
            day0 = np.datetime64(day, "D")
            for prn, g in g0.groupby("prn"):
                h = (g["time"].to_numpy() - day0) / np.timedelta64(1, "h")
                brk = np.where(np.diff(h) > 0.1)[0] + 1
                for hh, yy in zip(np.split(h, brk), np.split(g["stec"].to_numpy(), brk)):
                    ax.plot(hh, yy, color=c, lw=0.7, rasterized=True)
            if not many:
                handles.append(Line2D([], [], color=c, lw=1.5, label=day))
        if many:
            handles = [Line2D([], [], color=SAT_GRAY, lw=1, label="All days")]
        handles.append(Line2D([], [], color="none", label=f"each satellite, every {_fmt_s(dt)}"))
        _hour_axis(ax, station.utc_offset_h)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Universal time (h)", color=INK)
        ax.set_ylabel("Slant TEC (TECU)", color=INK)
        _title(ax, settings, f"{station.name.upper()}   {days[0]} to {days[-1]}   Slant TEC, elevation ≥ {el:g}°")
        _legend(fig, handles, ncol=1, loc="outside right upper")
    if return_fig:
        return fig
    return save(fig, path, settings)


# ───────────────────────────── flares and storms ─────────────────────────────
def _flare_shade(ax, row, peak):
    s = (row["start"] - peak).total_seconds() / 60
    e = (row["end"] - peak).total_seconds() / 60
    ax.axvspan(s, e, color="#fde7c2", lw=0, zorder=0)
    ax.axvline(0, color=ORANGE, lw=0.9, zorder=1)


def plot_flare(tracks, stack, row, name, el_min, settings, path, return_fig=False):
    """One flare: VTEC of each satellite, VTEC minus its pre-flare trend, and the rate of TEC,
    with the median over satellites, against minutes from the X-ray peak."""
    peak = pd.Timestamp(row["peak"])
    fig = _fig(settings, 0.95)
    with plt.rc_context(RC):
        axs = fig.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [1.2, 1.3, 1]})
        prns = sorted(tracks["prn"].unique())
        cols = {p: SERIES[k % len(SERIES)] for k, p in enumerate(prns)}
        for ax in axs:
            _style(ax)
            _flare_shade(ax, row, peak)
        for _, g in tracks.groupby("pass"):
            c = cols[g["prn"].iloc[0]]
            axs[0].plot(g["min_from_peak"], g["vtec"], color=c, lw=0.9)
            axs[1].plot(g["min_from_peak"], g["dvtec"], color=c, lw=0.6, alpha=0.55)
            axs[2].plot(g["min_from_peak"], g["rot"], color=c, lw=0.5, alpha=0.35)
        m = stack["min_from_peak"]
        many = stack["n_sats"] >= 2
        axs[1].fill_between(m, stack["dvtec_q1"].where(many), stack["dvtec_q3"].where(many),
                            color=BLUE_LIGHT, lw=0, alpha=0.7, zorder=2)
        axs[1].plot(m, stack["dvtec"].where(many), color=INK, lw=1.6, zorder=3)
        p99 = row.get("null_p99_tecu", np.nan)
        if np.isfinite(p99):
            axs[1].axhline(p99, color=RED, lw=0.8, ls=(0, (3, 2)), zorder=2)
        axs[1].axhline(0, color=INK2, lw=0.6)
        axs[2].plot(m, stack["rot"].where(many), color=INK, lw=1.4)
        axs[2].axhline(0, color=INK2, lw=0.6)
        lim = np.nanpercentile(np.abs(tracks["rot"]), 99) if tracks["rot"].notna().any() else 1
        axs[2].set_ylim(-1.2 * lim, 1.2 * lim)
        axs[0].set_ylabel("VTEC (TECU)", color=INK)
        axs[1].set_ylabel("VTEC minus\npre-flare trend (TECU)", color=INK)
        axs[2].set_ylabel("Rate of TEC\n(TECU/min)", color=INK)
        axs[2].set_xlim(-90, 90)
        axs[2].set_xlabel(f"Minutes from X-ray peak ({peak:%Y-%m-%d %H:%M} UT)", color=INK)
        ticks = np.arange(-90, 91, 30)
        sec = axs[0].secondary_xaxis("top")
        sec.set_xticks(ticks)
        sec.set_xticklabels([(peak + pd.Timedelta(minutes=float(t))).strftime("%H:%M") for t in ticks])
        sec.tick_params(colors=INK2, labelsize=7.5)
        for ax in axs:
            ax.set_xticks(ticks)
        lst = row["local_solar_time_h"]
        exc = row.get("excess_tecu", np.nan)
        if np.isfinite(exc) and "p_value" in row:
            verdict = (f"rise {exc:+.2f} TECU vs ordinary times {row['null_median_tecu']:+.2f} ± "
                       f"{row['null_robust_std_tecu']:.2f}, p = {row['p_value']:.3f} → "
                       + ("detected" if row["detected"] else "not detected"))
        else:
            verdict = row.get("note", "")
        _title(axs[0], settings,
               f"{name.upper()}   {row['class']} flare, {peak:%d %b %Y}   solar zenith {row['solar_zenith_deg']:.0f}°, "
               f"local solar time {int(lst):02d}:{int(lst % 1 * 60):02d}\n{verdict}")
        handles = [Line2D([], [], color=SERIES[0], lw=0.9,
                          label=f"Each satellite, elevation ≥ {el_min:g}° (30-s means of 1-s data)"),
                   Line2D([], [], color=INK, lw=1.6, label="Median over satellites"),
                   matplotlib.patches.Patch(color=BLUE_LIGHT, label="Interquartile range"),
                   Line2D([], [], color=RED, lw=0.8, ls=(0, (3, 2)), label="99th percentile at ordinary times"),
                   matplotlib.patches.Patch(color="#fde7c2", label="X-ray flare, start to end"),
                   Line2D([], [], color=ORANGE, lw=0.9, label="X-ray peak")]
        _legend(fig, handles, ncol=2)
        fig.align_ylabels(axs)
    if return_fig:
        return fig
    return save(fig, path, settings)


def plot_flare_overview(stacks, nulls, name, el_min, settings, path, return_fig=False, combined=None):
    """Every sunlit flare's TEC rise next to the same quantity measured at ordinary times."""
    n = len(stacks)
    fig = _fig(settings, 0.16 + 0.075 * n)
    with plt.rc_context(RC):
        ax = fig.add_subplot()
        _style(ax)
        ax.grid(False, axis="y")
        labels = []
        for k, ((row, _), null) in enumerate(zip(stacks, nulls)):
            y = n - 1 - k
            if null.size:
                q = np.quantile(null, [0.01, 0.25, 0.5, 0.75, 0.99])
                ax.plot([q[0], q[4]], [y, y], color=INK2, lw=0.9, zorder=2)
                ax.fill_betweenx([y - 0.22, y + 0.22], q[1], q[3], color=BLUE_LIGHT, lw=0, zorder=2)
                ax.plot([q[2], q[2]], [y - 0.22, y + 0.22], color=INK2, lw=1.2, zorder=3)
            ax.plot(row["excess_tecu"], y, "o", ms=6.5, mec=SURFACE, mew=0.8, zorder=4,
                    color=ORANGE if row.get("detected") else INK)
            p = pd.Timestamp(row["peak"])
            labels.append(f"{row['class']}  {p:%d %b %H:%M} UT  (SZA {row['solar_zenith_deg']:.0f}°)")
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels[::-1])
        ax.set_ylim(-0.7, n - 0.3)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_xlabel("VTEC rise beyond the pre-flare trend at the X-ray peak (TECU)", color=INK)
        text = (f"{name.upper()}   Flare-time TEC rise against ordinary times "
                f"(median of satellites ≥ {el_min:g}°)")
        if combined:
            text += f"\nAll {combined['n_flares']} sunlit flares combined (Fisher): p = {combined['fisher_p']:.2g}"
        _title(ax, settings, text)
        handles = [Line2D([], [], marker="o", ls="none", color=ORANGE, ms=6, label="Flare, p < 0.01"),
                   Line2D([], [], marker="o", ls="none", color=INK, ms=6, label="Flare, not significant"),
                   matplotlib.patches.Patch(color=BLUE_LIGHT, label="Ordinary times: middle 50 %"),
                   Line2D([], [], color=INK2, lw=0.9, label="Ordinary times: 1 to 99 %")]
        _legend(fig, handles, ncol=2)
    if return_fig:
        return fig
    return save(fig, path, settings)


def plot_storm(dev, dst, flares, quiet, name, settings, path, t0=None, t1=None, return_fig=False):
    """Station VTEC against the quiet-day median, the deviation (station and CODE map), and Dst."""
    dev = dev.sort_values("time")
    if t0 is not None:
        dev = dev[(dev["time"] >= t0) & (dev["time"] < t1)]
    tt = dev["time"]
    have_dst = dst is not None and len(dst)
    n = 3 if have_dst else 2
    fig = _fig(settings, 0.80 if have_dst else 0.6)
    with plt.rc_context(RC):
        axs = fig.subplots(n, 1, sharex=True, gridspec_kw={"height_ratios": [1.3, 1.1, 0.8][:n]})
        for ax in axs:
            _style(ax)
        # day boundaries, with gaps where a day is missing
        brk = dev["time"].diff() > pd.Timedelta(minutes=5)
        grp = brk.cumsum()

        def line(ax, y, **kw):
            for _, g in dev.assign(_y=y).groupby(grp):
                ax.plot(g["time"], g["_y"], **kw)
                kw.pop("label", None)

        for _, g in dev.groupby(grp):
            axs[0].fill_between(g["time"], g["lower_bound"], g["upper_bound"], color=BLUE_LIGHT, lw=0, alpha=0.8)
        line(axs[0], dev["quiet_median"], color=INK2, lw=0.9, ls=(0, (4, 2)))
        line(axs[0], dev["vtec"], color=BLUE, lw=1.2)
        axs[0].set_ylim(bottom=0)
        axs[0].set_ylabel("VTEC (TECU)", color=INK)
        pos = dev["dvtec"].where(dev["vtec"] > dev["upper_bound"])
        neg = dev["dvtec"].where(dev["vtec"] < dev["lower_bound"])
        axs[1].axhline(0, color=INK2, lw=0.6)
        line(axs[1], dev["dvtec"], color=BLUE, lw=0.9)
        for _, g in dev.assign(p=pos, q=neg).groupby(grp):
            axs[1].fill_between(g["time"], 0, g["p"].fillna(0), color=ORANGE, lw=0, alpha=0.6)
            axs[1].fill_between(g["time"], 0, g["q"].fillna(0), color="#6250d6", lw=0, alpha=0.5)
        have_gim = "gim_dvtec" in dev and dev["gim_dvtec"].notna().any()
        if have_gim:
            g = dev.dropna(subset=["gim_dvtec"])
            axs[1].plot(g["time"], g["gim_dvtec"], ls="none", marker="o", ms=2.2, color=INK2)
        axs[1].set_ylabel("ΔVTEC from quiet\nmedian (TECU)", color=INK)
        if have_dst:
            d = dst[(dst["time"] >= tt.min() - pd.Timedelta(hours=1)) & (dst["time"] <= tt.max() + pd.Timedelta(hours=1))]
            axs[2].plot(d["time"], d["dst"], color=INK, lw=1.0, drawstyle="steps-mid")
            axs[2].axhline(0, color=INK2, lw=0.6)
            axs[2].axhline(-50, color=RED, lw=0.6, ls=(0, (3, 2)))
            axs[2].set_ylabel("Dst (nT)", color=INK)
        if flares is not None and len(flares):
            f = flares[(flares["peak"] >= tt.min()) & (flares["peak"] <= tt.max())]
            for ax in axs:
                for p in f["peak"]:
                    ax.axvline(p, color=ORANGE, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
        days = pd.date_range(tt.min().normalize(), tt.max().normalize() + pd.Timedelta(days=1), freq="D")
        step = max(1, int(np.ceil(len(days) / 12)))
        axs[-1].set_xticks(days[::step])
        axs[-1].set_xticklabels([d.strftime("%m-%d") for d in days[::step]])
        axs[-1].set_xticks(days, minor=True)
        axs[-1].set_xlim(tt.min().normalize(), tt.max().normalize() + pd.Timedelta(days=1))
        axs[-1].set_xlabel("Date (UT)", color=INK)
        nq = int(dev["n_quiet_days"].median())
        _title(axs[0], settings, f"{name.upper()}   Storm-time VTEC against quiet days "
                                 f"(median of {nq} nearest quiet days at the same UT)")
        handles = [Line2D([], [], color=BLUE, lw=1.2, label="Station VTEC (gradient-corrected fit)"),
                   Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 2)), label="Quiet-day median"),
                   matplotlib.patches.Patch(color=BLUE_LIGHT, label="Quiet bounds: median ± 1.5 × IQR"),
                   matplotlib.patches.Patch(color=ORANGE, alpha=0.6, label="Above upper bound"),
                   matplotlib.patches.Patch(color="#6250d6", alpha=0.5, label="Below lower bound")]
        if have_gim:
            handles.append(Line2D([], [], ls="none", marker="o", ms=3, color=INK2,
                                  label="CODE map, same method (hourly maps)"))
        if flares is not None and len(flares):
            handles.append(Line2D([], [], color=ORANGE, lw=0.7, ls=(0, (1, 1.5)), label="Flare X-ray peak"))
        if have_dst:
            handles.append(Line2D([], [], color=INK, lw=1.0, label="Dst, WDC Kyoto (hourly)"))
        _legend(fig, handles, ncol=2)
        fig.align_ylabels(axs)
    if return_fig:
        return fig
    return save(fig, path, settings)
