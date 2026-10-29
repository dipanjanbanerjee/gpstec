import numpy as np

from gpstec.geometry import ecef_to_geodetic, geodetic_to_ecef, mapping_factor


def test_ecef_round_trip():
    xyz = geodetic_to_ecef(23.3607, 86.3408, 285.279)
    lat, lon, h = ecef_to_geodetic(np.asarray(xyz))[:3]
    assert abs(lat - 23.3607) < 1e-8 and abs(lon - 86.3408) < 1e-8 and abs(h - 285.279) < 1e-3


def test_mapping_factor_zenith_and_low():
    assert abs(mapping_factor(90.0, 350.0) - 1.0) < 1e-12
    assert 0.3 < mapping_factor(10.0, 350.0) < 0.4
