"""Physical and GPS constants used across the package."""

C = 299_792_458.0                 # speed of light (m/s)
F1 = 1575.42e6                    # GPS L1 (Hz)
F2 = 1227.60e6                    # GPS L2 (Hz)
F5 = 1176.45e6                    # GPS L5 (Hz)
LAMBDA1 = C / F1                  # L1 wavelength (m)
LAMBDA2 = C / F2                  # L2 wavelength (m)
LAMBDA5 = C / F5

# 1 TECU = 1e16 electrons / m^2.  Ionospheric group delay on frequency f is
# 40.3 * TEC / f^2 metres, so (P2 - P1) in metres converts to TECU with:
K_L1L2 = F1**2 * F2**2 / (F1**2 - F2**2) / 40.3 / 1e16    # ~9.52 TECU per metre
K_L1L5 = F1**2 * F5**2 / (F1**2 - F5**2) / 40.3 / 1e16    # ~7.76 TECU per metre
TECU_PER_NS_L1L2 = K_L1L2 * C * 1e-9                      # ~2.85 TECU per ns of DCB

# WGS-84
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
GM = 3.986005e14                  # IS-GPS-200 value
OMEGA_E = 7.2921151467e-5         # Earth rotation rate (rad/s)

RE_KM = 6371.0                    # mean Earth radius for the thin-shell model
