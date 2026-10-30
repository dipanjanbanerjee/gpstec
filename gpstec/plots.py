"""Publication-quality VTEC figures.

Every figure is laid out with matplotlib's constrained layout, legends sit outside
the data area, and each figure is written in all formats listed in
``settings.figure_formats`` (vector PDF/SVG/EPS and 300 dpi PNG/TIFF by default).
``text_overlaps`` lets the tests prove that no two pieces of text touch.
"""
from pathlib import Path

import numpy as np
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


def plot_day(d, series, station, settings, path, title_extra="", return_fig=False, gim_hourly=None):
    """One day, measured values only: every satellite's VTEC at the receiver's own
    sampling, the per-bin mean of the satellites, and the CODE map at its map epochs.
    Nothing is interpolated or smoothed between points."""
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
        m = series["vtec"].to_numpy()
        handles = [Line2D([], [], color=SAT_GRAY, lw=1.2,
                          label=f"Each satellite ≥ {el:g}°, every {_fmt_s(dt_rx)} measurement"),
                   Line2D([], [], color=BLUE, lw=1.4, label=f"Station VTEC, {binm:g}-min mean of the satellites")]
        if "vtec_uncertainty" in series and series["vtec_uncertainty"].notna().any():
            u = series["vtec_uncertainty"].to_numpy()
            ax.fill_between(x, m - u, m + u, color=BLUE_LIGHT, lw=0, zorder=2)
            handles.append(matplotlib.patches.Patch(color=BLUE_LIGHT,
                                                    label=f"Calibration uncertainty (±{np.nanmax(u):.1f} TECU)"))
        ax.plot(x, m, color=BLUE, lw=1.4, zorder=4)
        if getattr(settings, "smooth_curve", False) and "vtec_smooth" in series:
            ax.plot(x, series["vtec_smooth"], color=INK, lw=0.9, ls=(0, (1, 1.5)), zorder=5)
            handles.append(Line2D([], [], color=INK, lw=0.9, ls=(0, (1, 1.5)),
                                  label=f"Spline model (knots every {settings.smooth_knot_min:g} min)"))
        gx, gy, gattr = _gim_points(series, gim_hourly)
        if gx is not None:
            ax.plot(gx, gy, "D", ms=3.6, color=ORANGE, mec=SURFACE, mew=0.6, ls="none", zorder=6)
            handles.append(Line2D([], [], marker="D", ms=4, color=ORANGE, mec=SURFACE, ls="none",
                                  label=_gim_label(gattr)))

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
            ax.annotate(f"{m[i]:.1f} TECU, {x[i]:.2f} UT", (x[i], m[i]), xytext=(6, 5),
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
            ax.plot(s["ut_hour"] + settings.bin_minutes / 120.0, s["vtec"], color=color,
                    lw=1.0 if many else 1.5)
            if not many:
                handles.append(Line2D([], [], color=color, lw=1.5, label=day))
        if many:
            piv = all_series.pivot_table(index="ut_hour", columns="date", values="vtec")
            ax.plot(piv.index, piv.mean(axis=1), color=BLUE, lw=2)
            handles = [Line2D([], [], color=SAT_GRAY, lw=1, label="Individual days"),
                       Line2D([], [], color=BLUE, lw=2, label=f"Mean of {len(days)} days")]
        _hour_axis(ax, station.utc_offset_h)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Universal time (h)", color=INK)
        ax.set_ylabel("VTEC (TECU)", color=INK)
        _title(ax, settings, f"{station.name.upper()}   {days[0]} to {days[-1]}   Vertical TEC, elevation ≥ "
                             f"{settings.output_elevation:g}°, {settings.bin_minutes:g}-min means")
        _legend(fig, handles, ncol=1, loc="outside right upper")
    if return_fig:
        return fig
    return save(fig, path, settings)


def plot_heatmap(all_series, station, settings, path, return_fig=False):
    """Day × UT map of station VTEC (gaps blank)."""
    piv = all_series.pivot_table(index="date", columns="ut_hour", values="vtec", dropna=False)
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
        _title(ax, settings, f"{station.name.upper()}   Vertical TEC by day, elevation ≥ {settings.output_elevation:g}°, "
                             f"{settings.bin_minutes:g}-min bins")
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
        _title(ax, settings, f"{station.name.upper()}   {day}   Pierce points at {settings.shell_height_km:g} km, every {_fmt_s(_interval_s(s["time"]))}")
    if return_fig:
        return fig
    return save(fig, path, settings)


def plot_continuity(all_series, rows, station, settings, path, return_fig=False):
    """All days on one continuous time axis, plus the VTEC jump found at each midnight."""
    days = sorted(all_series["date"].astype(str).unique())
    nd = len(days)
    d0 = np.datetime64(days[0])
    step = max(1, int(np.ceil(nd / 10)))
    fig = _fig(settings, 0.62)
    with plt.rc_context(RC):
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.25])
        ax = fig.add_subplot(gs[0])
        bx = fig.add_subplot(gs[1])
        _style(ax)
        _style(bx)
        have_gim = "vtec_gim" in all_series and all_series["vtec_gim"].notna().any()
        for day in days:
            s = all_series[all_series["date"].astype(str) == day]
            x = (np.datetime64(day) - d0).astype("timedelta64[h]").astype(float) + s["ut_hour"].to_numpy()
            ax.plot(x, s["vtec"], color=BLUE, lw=1.0, zorder=3)
            if have_gim and "vtec_gim" in s and s["vtec_gim"].notna().any():
                g = s["vtec_gim"].notna().to_numpy()
                ax.plot(x[g], s["vtec_gim"].to_numpy()[g], "D", ms=2.6, color=ORANGE, mec=SURFACE, mew=0.4,
                        ls="none", zorder=4)
        for k in range(1, nd):
            for a in (ax, bx):
                a.axvline(24 * k, color=INK2, lw=0.5, ls=(0, (1, 2)), zorder=1)
        ax.set_xlim(0, 24 * nd)
        ax.set_ylim(bottom=0)
        ax.set_xticks([24 * k + 12 for k in range(0, nd, step)])
        ax.set_xticks([24 * k for k in range(nd + 1)], minor=True)
        ax.tick_params(axis="x", which="major", length=0)
        ax.set_xticklabels([days[k][5:] for k in range(0, nd, step)])
        ax.grid(False, axis="x")
        ax.set_ylabel("VTEC (TECU)", color=INK)
        _title(ax, settings, f"{station.name.upper()}   Day-to-day continuity, elevation ≥ {settings.output_elevation:g}°")

        xs, js, tol, col = [], [], [], []
        for r in rows:
            if r.get("jump_median_el30") is None or r["next_date"] not in days:
                continue
            xs.append(24 * days.index(r["next_date"]))
            js.append(r["jump_median_el30"])
            tol.append(r["tolerance_tecu"])
            col.append(BLUE if r["status"] == "OK" else RED)
        for x, t in zip(xs, tol):
            bx.fill_between([x - 2.5, x + 2.5], -t, t, color=BLUE_LIGHT, lw=0, zorder=1)
        bx.scatter(xs, js, s=22, c=col, edgecolors=SURFACE, linewidths=0.8, zorder=3)
        if nd <= 12:
            for x, j in zip(xs, js):
                bx.annotate(f"{j:+.1f}", (x, j), xytext=(7, 0), textcoords="offset points", va="center",
                            fontsize=7.5, color=INK, zorder=4)
        lim = max(3.0, 1.35 * max([abs(v) for v in js] + tol)) if js else 3.0
        bx.set_ylim(-lim, lim)
        bx.axhline(0, color=INK2, lw=0.6)
        bx.yaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
        bx.set_ylabel("Jump\n(TECU)", color=INK)
        bx.set_xticks([24 * k for k in range(1, nd, step)])
        bx.set_xticklabels([days[k][5:] for k in range(1, nd, step)])
        bx.set_xlabel("Midnight UT (date of the new day)", color=INK)
        bx.grid(False, axis="x")
        bx.set_xlim(ax.get_xlim())
        handles = [Line2D([], [], color=BLUE, lw=1.0, label=f"Station VTEC, {settings.bin_minutes:g}-min means")]
        if have_gim:
            handles.append(Line2D([], [], marker="D", ms=3.5, color=ORANGE, mec=SURFACE, ls="none",
                                  label="CODE global map, hourly maps"))
        handles += [Line2D([], [], marker="o", ls="none", color=BLUE, ms=5, label="Midnight jump, within uncertainty"),
                    Line2D([], [], marker="o", ls="none", color=RED, ms=5, label="Midnight jump, outside it"),
                    matplotlib.patches.Patch(color=BLUE_LIGHT, label="Calibration uncertainty")]
        _legend(fig, handles, ncol=3)
        fig.align_ylabels([ax, bx])
    if return_fig:
        return fig
    return save(fig, path, settings)
