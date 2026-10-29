# Changelog

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
