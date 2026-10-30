# Changelog

## 1.1.0 (2026-10-30)

- New slant TEC figures: one per day (every satellite at 1 s, with its elevation), an overlay of all days, and a continuity figure with the STEC jump at each midnight.
- The continuity figure is redrawn: colours alternate by day, dotted lines mark midnight UT, and the lower panel shows the jump at each midnight against the calibration uncertainty.
- CODE GIM uses only its published hourly maps (25 per day). They are drawn as dots joined by a dashed line, and the new `STAT_DATE_gim_hourly.csv` stores them. The `vtec_gim` column of the 1-min CSV is filled only at hh:00. The GIM receiver-bias check compares data within 2.5 min of a map epoch. New option `gim_station = "bilinear" | "nearest"`.
- Legends, titles and the README state the time interval of every series. The receiver sampling interval is recorded in the log and in the calibration report.
- `smooth_curve = true` (default) keeps the gradient-corrected station curve. Set it to `false` to use the plain 1-min mean instead.

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
