"""Coordinates, look angles, pierce points and the thin-shell mapping function."""
import numpy as np

from .constants import WGS84_A, WGS84_E2, RE_KM


def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.array([
        (n + h_m) * np.cos(lat) * np.cos(lon),
        (n + h_m) * np.cos(lat) * np.sin(lon),
        (n * (1 - WGS84_E2) + h_m) * np.sin(lat),
    ])


def ecef_to_geodetic(xyz):
    x, y, z = xyz
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(6):
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - WGS84_E2 * n / (n + h)))
    return np.degrees(lat), np.degrees(lon), h


def enu_matrix(lat_deg, lon_deg):
    """Rows are the E, N, U unit vectors expressed in ECEF."""
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([
        [-np.sin(lo), np.cos(lo), 0.0],
        [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)],
    ])


def az_el(station_ecef, lat_deg, lon_deg, sat_ecef):
    """Azimuth and elevation (degrees) of satellites (N x 3 ECEF) from a station."""
    d = np.atleast_2d(sat_ecef) - station_ecef
    enu = d @ enu_matrix(lat_deg, lon_deg).T
    az = np.degrees(np.arctan2(enu[:, 0], enu[:, 1])) % 360.0
    el = np.degrees(np.arctan2(enu[:, 2], np.hypot(enu[:, 0], enu[:, 1])))
    return az, el


def mapping_factor(el_deg, shell_km, re_km=RE_KM):
    """cos(z') of the thin-shell model: VTEC = STEC * mapping_factor."""
    sin_z = re_km / (re_km + shell_km) * np.cos(np.radians(el_deg))
    return np.sqrt(1.0 - sin_z ** 2)


def pierce_point(lat_deg, lon_deg, az_deg, el_deg, shell_km, re_km=RE_KM):
    """Ionospheric pierce point latitude/longitude (degrees) on a spherical shell."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    az, el = np.radians(az_deg), np.radians(el_deg)
    psi = np.pi / 2 - el - np.arcsin(re_km / (re_km + shell_km) * np.cos(el))
    ipp_lat = np.arcsin(np.sin(lat) * np.cos(psi) + np.cos(lat) * np.sin(psi) * np.cos(az))
    ipp_lon = lon + np.arcsin(np.clip(np.sin(psi) * np.sin(az) / np.cos(ipp_lat), -1, 1))
    return np.degrees(ipp_lat), (np.degrees(ipp_lon) + 180) % 360 - 180
