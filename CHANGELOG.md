# Changelog

## 1.1.0 (2026-10-30)

- Plots show measured values only. The grey lines are every 1 s value of each satellite, the blue line is the 1-min mean of the satellites, and CODE GIM points are drawn only at its hourly map epochs. Nothing is interpolated in time.
- The gradient-corrected spline curve is now optional (`smooth_curve = true`) and off by default. The `.Std` file, peak VTEC and the overlay, heatmap and continuity figures use the 1-min means.
- New `STAT_DATE_gim_hourly.csv`. The `vtec_gim` column of the 1-min CSV is filled only at hh:00.
- New `gim_station = "bilinear" | "nearest"`.
- Legends and titles state the sampling interval of each series. The README has a time and space resolution table.
- The run log and calibration report record the receiver sampling interval.

## 1.0.0 (2026-10-29)

First public release.

- `.GPS` (NovAtel GPStation-6) or RINEX 2/3 input → calibrated STEC/VTEC with a configurable elevation cut-off (50° default).
- Station position always taken from the configuration (RINEX headers from RTKCONV 2.4.2 are wrong).
- Satellite biases from CODE DCB or Bias-SINEX files, downloaded automatically.
- Per-day receiver bias from night-time least squares with horizontal gradients; never allows negative VTEC.
- Gradient-corrected smooth station VTEC curve.
- Satellite passes that cross midnight are levelled as one; automatic midnight continuity check.
- CODE global ionosphere map as an independent reference on every plot and in the calibration report.
- Publication-quality figures (PDF + 300 dpi PNG), checked for overlapping text.
- Commands: `gpstec init`, `gpstec run`, `gpstec install-convbin`, `gpstec --version`.
