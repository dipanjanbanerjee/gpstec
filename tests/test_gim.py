import gzip

import numpy as np

from gpstec import gim


def _write_ionex(path, value_tecu=25.0):
    lines = ["     1.0            IONOSPHERE MAPS     GPS                 IONEX VERSION / TYPE",
             "    -1                                                      EXPONENT",
             "                                                            END OF HEADER"]
    lats = np.arange(87.5, -87.6, -2.5)
    for k, hh in enumerate((0, 1, 2)):
        lines.append(f"{k + 1:6d}                                                      START OF TEC MAP")
        lines.append(f"  2022    12    25    {hh:2d}     0     0                        EPOCH OF CURRENT MAP")
        for la in lats:
            lines.append(f"  {la:6.1f}-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H")
            vals = [int(value_tecu * 10)] * 73
            for i in range(0, 73, 16):
                lines.append("".join(f"{v:5d}" for v in vals[i:i + 16]))
        lines.append(f"{k + 1:6d}                                                      END OF TEC MAP")
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode()))


def test_read_ionex_and_station_series(tmp_path):
    p = tmp_path / "COD0OPSFIN_20223590000_01D_01H_GIM.INX.gz"
    _write_ionex(p, 25.0)
    ep, lats, lons, tec = gim.read_ionex(p)
    assert tec.shape == (3, len(lats), 73)
    t = np.array(["2022-12-25T00:30", "2022-12-25T01:45"], dtype="datetime64[ns]")
    v = gim.station_series(p, 23.36, 86.34, t)
    assert np.allclose(v, 25.0)
    assert gim.local_files_for("2022-12-25", tmp_path) == [p]
