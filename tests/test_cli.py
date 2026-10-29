import pytest

from gpstec import __author__, __main__ as cli
from gpstec.config import load


def test_version_names_author(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __author__ in capsys.readouterr().out


def test_init_writes_loadable_template(tmp_path):
    p = tmp_path / "station.toml"
    assert cli.main(["init", str(p)]) == 0
    st, se = load(p)
    assert se.output_elevation == 50.0 and st.lat is not None


def test_missing_station_position_is_an_error(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text("[station]\nname = 'abcd'\n")
    with pytest.raises(ValueError):
        load(p)
