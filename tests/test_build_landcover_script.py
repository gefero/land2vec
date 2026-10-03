"""Tests de scripts/datos/build_landcover_nc.py: elección de fuentes, CLI y --compare."""
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from _synth import LAT0, LON0, N, RES, _codes, _raw

_spec = importlib.util.spec_from_file_location("build_landcover_nc", Path(__file__).resolve().parents[1] / "scripts" / "datos" / "build_landcover_nc.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

BOX = (LON0, LAT0 - N * RES, LON0 + N * RES, LAT0)


def test_find_sources_elige_el_mas_chico_si_los_duplicados_son_identicos(tmp_path, capsys):
    chico = _raw(tmp_path / "recorte" / "2000_clip.nc", 2000, _codes()) if (tmp_path / "recorte").mkdir() is None else None
    grande = _raw(tmp_path / "global_2000.nc", 2000, _codes(), relleno=5000)
    assert grande.stat().st_size > chico.stat().st_size
    assert S.find_sources(tmp_path, [2000], BOX) == {2000: chico}
    assert "es idéntico" in capsys.readouterr().out


def test_find_sources_se_frena_si_los_duplicados_difieren(tmp_path):
    _raw(tmp_path / "a.nc", 2000, _codes())
    otro = _codes(); otro[5, 5] = 10                      # un solo píxel distinto
    _raw(tmp_path / "b.nc", 2000, otro, relleno=100)
    with pytest.raises(SystemExit, match="difieren en 1 px"):
        S.find_sources(tmp_path, [2000], BOX)


def test_find_sources_descarta_los_que_no_cubren_el_vector(tmp_path, capsys):
    bueno = _raw(tmp_path / "bueno.nc", 2000, _codes())
    _raw(tmp_path / "corrido.nc", 2000, _codes(), lon0=LON0 + 10 * RES)     # le falta la mitad oeste
    assert S.find_sources(tmp_path, [2000], BOX) == {2000: bueno}
    assert "descartado (no cubre el vector)" in capsys.readouterr().out


def test_find_sources_error_si_falta_un_año_y_si_el_unico_no_cubre(tmp_path):
    _raw(tmp_path / "a.nc", 2000, _codes())
    with pytest.raises(SystemExit, match=r"faltan años: \[2001\]"):
        S.find_sources(tmp_path, [2000, 2001], BOX)
    _raw(tmp_path / "b.nc", 2001, _codes(), lon0=LON0 + 10 * RES)
    with pytest.raises(SystemExit, match=r"faltan años: \[2001\]"):
        S.find_sources(tmp_path, [2000, 2001], BOX)


def test_find_sources_ignora_archivos_rotos_y_descomprime_zips(tmp_path, capsys):
    (tmp_path / "a medio copiar.nc").write_bytes(b"no soy un netcdf")
    src = _raw(tmp_path / "pila" / "x.nc", 2000, _codes()) if (tmp_path / "pila").mkdir() is None else None
    with zipfile.ZipFile(tmp_path / "descarga.zip", "w") as z:
        z.write(src, "2000_del_cds.nc")
    src.unlink(); src.parent.rmdir()
    got = S.find_sources(tmp_path, [2000], BOX)
    assert got[2000].parent.name == "_extraidos" and "se ignora" in capsys.readouterr().out


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["build_landcover_nc.py", *map(str, args)])


def test_cli_con_mapping_json_y_vector_como_caja(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    for y in (2000, 2001):
        _raw(raw / f"{y}.nc", y, _codes())
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"0": "Nd", "10": "Agr", "50": "Fo"}))
    out = tmp_path / "serie.nc"
    _argv(monkeypatch, "--raw-dir", raw, "--out", out, "--anual-dir", tmp_path / "anual", "--years", "2000-2001",
          f"--vector={','.join(map(str, BOX))}", "--mapping", mapping)
    S.main()
    with xr.open_dataset(out, mask_and_scale=False) as ds:
        assert json.loads(ds.attrs["estados"]) == {"0": "Nd", "1": "Agr", "2": "Fo"}
        assert ds.sizes["time"] == 2 and "bbox" in ds.attrs["vector"]
    assert sorted(p.name for p in (tmp_path / "anual").iterdir()) == ["lccs_2000.nc", "lccs_2001.nc"]


def test_cli_compare_detecta_un_pixel_distinto(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "raw"; raw.mkdir()
    for y in (2000, 2001):
        _raw(raw / f"{y}.nc", y, _codes())
    a, b = tmp_path / "a.nc", tmp_path / "b.nc"
    for out in (a, b):
        _argv(monkeypatch, "--raw-dir", raw, "--out", out, "--anual-dir", tmp_path / f"anual_{out.stem}",
              "--years", "2000-2001", f"--vector={','.join(map(str, BOX))}")
        S.main()
    _argv(monkeypatch, "--out", b, "--compare", a)
    S.main()
    assert "iguales 100.0000%" in capsys.readouterr().out
    # b con un píxel cambiado en 2001
    with xr.open_dataset(b) as d:
        d = d.load()
    d["lccs_class"].values[1, 3, 3] = 9
    d.to_netcdf(tmp_path / "c.nc")
    _argv(monkeypatch, "--out", tmp_path / "c.nc", "--compare", a)
    S.main()
    out = capsys.readouterr().out
    assert "difieren          1 px" in out
