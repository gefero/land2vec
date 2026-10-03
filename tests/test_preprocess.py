"""Tests sintéticos de land2vec.preprocess: crudos chicos con la estructura del CDS."""
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Polygon

from _synth import LAT0, LON0, N, RES, _codes, _raw
from land2vec import preprocess as pp

MAPPING = {0: "Nd", 10: "Agr", 50: "Fo"}


def test_read_raw_toma_año_y_versión_de_los_metadatos(tmp_path):
    p = _raw(tmp_path / "cualquier_nombre.nc", 2016, _codes(), version="2.1.1")
    da = pp.read_raw(p)
    assert (da.attrs["year"], da.attrs["product_version"]) == (2016, "2.1.1")
    assert da.dims == ("lat", "lon")


def test_group_codes_subniveles_y_tokens_del_dict(tmp_path):
    da = pp.read_raw(_raw(tmp_path / "a.nc", 2000, _codes()))
    g = pp.group_codes(da, MAPPING)
    assert g.attrs["estados"] == '{"0": "Nd", "1": "Agr", "2": "Fo"}'
    assert set(np.unique(g.values)) == {0, 1, 2}        # 11 hereda el token de 10
    assert g.values[5, 0] == 2 and g.values[5, N - 1] == 1


def test_group_codes_error_con_codigo_sin_token(tmp_path):
    da = pp.read_raw(_raw(tmp_path / "a.nc", 2000, _codes(shift=100)))  # 111 -> padre 110: sin token
    with pytest.raises(ValueError, match="sin token"):
        pp.group_codes(da, MAPPING)


def test_default_reproduce_el_orden_del_netcdf_original():
    assert pp.token_table(pp.LCCS_IPCC) == ["Nd", "A", "F", "G", "Wt", "U", "Sh", "Sp", "B", "Wa"]


def test_clip_con_poligono_deja_255_afuera(tmp_path):
    da = pp.read_raw(_raw(tmp_path / "a.nc", 2000, _codes()))
    # triángulo en la mitad izquierda de la grilla
    tri = Polygon([(LON0, LAT0), (LON0 + 10 * RES, LAT0), (LON0, LAT0 - 10 * RES)])
    out = pp.clip_to_vector(da, tri)
    assert out.shape == (10, 10)                          # recorte al bbox del polígono
    assert (out.values == pp.FILL).any() and (out.values != pp.FILL).any()
    assert out.values[0, 0] == 0 and out.values[9, 9] == pp.FILL   # esquina opuesta, fuera del triángulo


def test_clip_con_caja_no_enmascara_y_exige_cobertura(tmp_path):
    da = pp.read_raw(_raw(tmp_path / "a.nc", 2000, _codes()))
    out = pp.clip_to_vector(da, (LON0, LAT0 - 10 * RES, LON0 + 10 * RES, LAT0))
    assert out.shape == (10, 10) and not (out.values == pp.FILL).any()
    with pytest.raises(ValueError, match="no abarca"):
        pp.clip_to_vector(da, (LON0, LAT0 - 50 * RES, LON0 + 10 * RES, LAT0))


def test_ciclo_completo_y_secuencia_Fo_Agr(tmp_path):
    tri = Polygon([(LON0, LAT0), (LON0 + 20 * RES, LAT0), (LON0, LAT0 - 20 * RES)])
    files = []
    for y in range(2000, 2003):
        # el año 2002 cambia: la mitad derecha pasa de cultivo a bosque
        codes = _codes() if y < 2002 else np.where(_codes() == 11, 50, _codes()).astype(np.uint8)
        files.append(pp.process_year(_raw(tmp_path / f"r{y}.nc", y, codes), tri, tmp_path / "anual", MAPPING))
    assert [f.name for f in files] == ["lccs_2000.nc", "lccs_2001.nc", "lccs_2002.nc"]
    # el anual guarda sólo la variable agrupada
    with xr.open_dataset(files[0], mask_and_scale=False) as ds:
        assert list(ds.data_vars) == ["lccs_class"] and ds.lccs_class.dtype == np.uint8

    serie = pp.stack_years(files[::-1], tmp_path / "serie.nc")          # desordenados a propósito
    with xr.open_dataset(serie) as ds:
        assert ds.sizes["time"] == 3 and list(ds.time.dt.year.values) == [2000, 2001, 2002]
        tab = pp.sequences_table(ds)
    assert not tab.seqs.str.contains("255").any() and (tab.seqs.str.count("-") == 2).all()
    assert set(tab.seqs) >= {"Fo-Fo-Fo", "Agr-Agr-Fo"}
    assert len(tab) < N * N                                              # los px fuera del triángulo no están
    assert tab.ID.is_unique and tab.ID.max() < N * N
    # con drop_fill=False los px de afuera vuelven como Nd
    with xr.open_dataset(serie) as ds:
        assert len(pp.sequences_table(ds, drop_fill=False)) == ds.sizes["lat"] * ds.sizes["lon"]


def test_write_sequences_zip_y_parquet_coinciden(tmp_path):
    import pandas as pd
    files = [pp.process_year(_raw(tmp_path / f"r{y}.nc", y, _codes()), (LON0, LAT0 - N * RES, LON0 + N * RES, LAT0),
                             tmp_path / "anual", MAPPING) for y in (2000, 2001)]
    serie = pp.stack_years(files, tmp_path / "s.nc")
    with xr.open_dataset(serie) as ds:
        ref = pp.sequences_table(ds, lat_rows=7)
        pp.write_sequences(ds, tmp_path / "t.zip", lat_rows=7)
        pp.write_sequences(ds, tmp_path / "t.parquet", lat_rows=7)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "t.zip"), ref, check_dtype=False)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "t.parquet"), ref, check_dtype=False)


def test_stack_years_rechaza_huecos_y_dict_distinto(tmp_path):
    box = (LON0, LAT0 - N * RES, LON0 + N * RES, LAT0)
    f = {y: pp.process_year(_raw(tmp_path / f"r{y}.nc", y, _codes()), box, tmp_path / "a", MAPPING) for y in (2000, 2001, 2003)}
    with pytest.raises(ValueError, match="faltan años"):
        pp.stack_years(list(f.values()), tmp_path / "x.nc")
    otro = pp.process_year(_raw(tmp_path / "r2002.nc", 2002, _codes()), box, tmp_path / "b", {**MAPPING, 50: "Bosque"})
    with pytest.raises(ValueError, match="distinto"):
        pp.stack_years([f[2000], f[2001], otro], tmp_path / "x.nc")


# ----------------------------------------------------------------------------- vectores
BOX = (LON0, LAT0 - N * RES, LON0 + N * RES, LAT0)


def _gdf(crs):
    import geopandas as gpd
    from shapely.geometry import box
    g = gpd.GeoDataFrame(geometry=[box(*BOX)], crs=4326)
    return g if crs == 4326 else g.to_crs(crs)


@pytest.mark.parametrize("ext", [".geojson", ".shp"])
def test_as_geoseries_lee_archivos_vectoriales(tmp_path, ext):
    _gdf(4326).to_file(tmp_path / f"v{ext}")
    g = pp.as_geoseries(tmp_path / f"v{ext}")
    assert np.allclose(g.total_bounds, BOX, atol=1e-9) and g.crs.to_epsg() == 4326


def test_as_geoseries_reproyecta_y_acepta_todas_las_formas():
    from shapely.geometry import box
    for v in (_gdf(3857), _gdf(3857).geometry):                       # GeoDataFrame y GeoSeries en otro CRS
        assert np.allclose(pp.as_geoseries(v).total_bounds, BOX, atol=1e-6)
    for v in (BOX, list(BOX), box(*BOX)):                             # tupla, lista, geometría: se asume 4326
        assert np.allclose(pp.as_geoseries(v).total_bounds, BOX)
    sin_crs = _gdf(4326).set_crs(None, allow_override=True)
    assert pp.as_geoseries(sin_crs).crs.to_epsg() == 4326
    with pytest.raises(TypeError):
        pp.as_geoseries(42)


def test_clip_da_lo_mismo_con_un_vector_en_otro_crs(tmp_path):
    da = pp.read_raw(_raw(tmp_path / "a.nc", 2000, _codes()))
    # catetos de 10,3 px: la hipotenusa no pasa por ningún centro de píxel, así que el redondeo de la proyección no la decide
    tri = Polygon([(LON0, LAT0), (LON0 + 10.3 * RES, LAT0), (LON0, LAT0 - 10.3 * RES)])
    import geopandas as gpd
    a = pp.clip_to_vector(da, tri)
    b = pp.clip_to_vector(da, gpd.GeoSeries([tri], crs=4326).to_crs(3857))
    assert a.shape == b.shape == (10, 10) and np.array_equal(a.values, b.values)
    assert (a.values == pp.FILL).any() and (a.values != pp.FILL).any()


def test_vector_mask_no_acumula_deriva_con_coordenadas_float32():
    """Regresión del censo: usar lon[1]-lon[0] de coordenadas float32 como paso corría el borde
    varios px al final de la grilla. Se compara con la geometría exacta sobre una grilla grande."""
    import shapely
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    n = 3000
    exact_lat = LAT0 - (np.arange(n) + 0.5) * RES
    exact_lon = LON0 + (np.arange(n) + 0.5) * RES
    lat, lon = exact_lat.astype(np.float32), exact_lon.astype(np.float32)
    poly = Polygon([(LON0 + 100 * RES, LAT0), (LON0 + n * RES, LAT0 - 50 * RES), (LON0 + 2400 * RES, LAT0 - n * RES)])
    xx, yy = np.meshgrid(exact_lon, exact_lat)
    exact = shapely.contains_xy(poly, xx, yy)
    mask = pp.vector_mask(lat, lon, poly)
    dx, dy = float(lon[1] - lon[0]), float(lat[0] - lat[1])           # el método anterior
    naive = rasterize([(poly, 1)], out_shape=(n, n), transform=from_origin(float(lon[0]) - dx / 2, float(lat[0]) + dy / 2, dx, dy),
                      fill=0, dtype="uint8").astype(bool)
    assert (mask != exact).sum() <= 20
    assert (naive != exact).sum() > 100 * max(1, (mask != exact).sum())  # el test detecta el defecto


def test_vector_mask_exige_orientacion_de_los_ejes():
    with pytest.raises(ValueError, match="lat descendente"):
        pp.vector_mask(np.arange(5.0), np.arange(5.0), BOX)


# ----------------------------------------------------------------------------- netCDF original
def test_sequences_table_con_el_netcdf_original_sin_atributos(tmp_path):
    "El .nc original trae uint8 con _FillValue 255 (xarray lo abre como float32 con NaN) y no tiene 'estados'."
    arr = np.full((2, 3, 4), 2, np.uint8)
    arr[1, :, 2:] = 1                   # F -> A en las dos columnas de la derecha
    arr[:, 0, 0] = 255                  # un px sin dato
    ds = xr.Dataset({"lccs_class": (("time", "lat", "lon"), arr)},
                    coords={"time": [0, 366], "lat": [-1.0, -1.1, -1.2], "lon": [5.0, 5.1, 5.2, 5.3]})
    ds["lccs_class"].encoding.update(_FillValue=np.uint8(255), dtype="uint8")
    ds.to_netcdf(tmp_path / "viejo.nc")
    with xr.open_dataset(tmp_path / "viejo.nc") as d:
        assert d["lccs_class"].dtype == np.float32                  # NaN en lugar de 255
        tab = pp.sequences_table(d)
    assert len(tab) == 11 and 0 not in tab.ID.values               # el px sin dato no está
    assert set(tab.seqs) == {"F-F", "F-A"}                          # tokens por defecto: 2 = F, 1 = A
    assert (tab.seqs == "F-A").sum() == 6 and (tab.seqs == "F-F").sum() == 5    # 3 filas x 2 columnas cambian
