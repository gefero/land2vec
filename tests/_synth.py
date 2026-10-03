"""Mapas crudos sintéticos con la estructura del CDS, compartidos por los tests."""
import h5netcdf.legacyapi as nc
import numpy as np

RES = 1 / 360
N = 20                       # grilla cruda de N x N px
LAT0, LON0 = -30.0, -60.0    # borde superior izquierdo


def _raw(path, year, codes, version="2.0.7cds", lat0=LAT0, lon0=LON0, relleno=0):
    """Crudo tipo CDS: int8 con _Unsigned, lat descendente, 1 paso de tiempo y una variable auxiliar.
    `relleno` agrega esa cantidad de bytes incompresibles, para que un archivo pese más que otro."""
    lat = lat0 - (np.arange(N) + 0.5) * RES
    lon = lon0 + (np.arange(N) + 0.5) * RES
    with nc.Dataset(path, "w") as f:
        f.createDimension("time", 1); f.createDimension("lat", N); f.createDimension("lon", N)
        f.createVariable("lat", "f8", ("lat",))[:] = lat
        f.createVariable("lon", "f8", ("lon",))[:] = lon
        v = f.createVariable("lccs_class", "i1", ("time", "lat", "lon"))
        v.setncattr("_Unsigned", "true")
        v[0] = codes.astype(np.uint8).view(np.int8)
        f.createVariable("change_count", "u1", ("time", "lat", "lon"))[0] = 0
        if relleno:
            f.createDimension("r", relleno)
            f.createVariable("relleno", "u1", ("r",))[:] = np.random.default_rng(0).integers(0, 256, relleno, dtype=np.uint8)
        f.setncattr("id", f"ESACCI-LC-L4-LCCS-Map-300m-P1Y-{year}-v{version}")
        f.setncattr("time_coverage_start", f"{year}0101")
        f.setncattr("product_version", version)
    return path


def _codes(shift=0):
    c = np.full((N, N), 50, np.uint8)       # bosque
    c[:, N // 2:] = 11 + shift              # cultivo con subnivel (11) a la derecha
    c[0, 0] = 0                             # un píxel no data
    return c
