"""Cruza los polígonos de desmonte de Colección 13.0 (Chaco Seco,
monitoreodesmonte.com.ar, `data/geo/data_validacion_chaco_Coleccion_13.0.rar`)
contra la grilla de píxeles ESA CCI de 300 m de cada zona, vía `land2vec.geo`.
Ver `docs/v2/paper_metodologia.md` §5.8 para el detalle metodológico completo.

No toca nada de `land2vec.cluster`: es un preprocesamiento puramente
geoespacial. Su salida se cruza después, por `(zone, ID)`, contra
`data/v2/clusters_train_pooled{suffix}_full.zip` (ver
`scripts/clustering/assign_train_clusters.py --group all --max-constant-fraction 1.0
--out-tag _full`) en `scripts/validacion/eval_desmonte.py`.

Prerrequisitos:
    data/geo/data_validacion_chaco_Coleccion_13.0.rar   el shapefile (es un ZIP, no se extrae)
    data/zonas/lat_long_df_<zona>.zip                          coordenadas por píxel de la zona

Escribe, por zona:
    data/desmonte/desmonte_px_epoca_<zona>.zip   tabla larga, fuente de verdad: ID,anio,frac,n_pol
                                          (anio = -1 previo a 2001, año 2001-2022, 9999 posterior)
    data/desmonte/desmonte_px_<zona>.zip         resumen, una fila por cada ID de lat_long_df_<zona>.zip:
                                          ID,frac_previo,frac_ventana,frac_post,anio_dom,frac_dom,
                                          n_epocas,n_pol,relevado
    data/desmonte/desmonte_poly_px_<zona>.zip    tabla indexada por polígono, sin agregar:
                                          poly_id,ID,anio,frac,SUPERF_ha -- la consume la métrica
                                          de detección por polígono de scripts/validacion/eval_desmonte.py

Uso:
    python scripts/datos/build_desmonte_labels.py --inspect
    python scripts/datos/build_desmonte_labels.py --zone periurbano_cordoba          # la zona chica primero
    python scripts/datos/build_desmonte_labels.py --zone chaco_santiago_frontier --zone yungas
    python scripts/datos/build_desmonte_labels.py                                    # las 3 zonas con polígonos
    python scripts/datos/build_desmonte_labels.py --zone periurbano_cordoba --check-methods
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from land2vec import paths as P  # noqa: E402
from land2vec.paths import rel  # noqa: E402

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402

from land2vec import geo as G  # noqa: E402
from land2vec.zones import CHACO_ZONE, EVAL_ZONES, TRAIN_ZONES  # noqa: E402

DATA_DIR = ROOT / "data"
GEO_DIR = DATA_DIR / "geo"
RAR_PATH = GEO_DIR / "data_validacion_chaco_Coleccion_13.0.rar"

# mismo criterio que land2vec.zones -- sin duplicar los bbox acá, ver src/land2vec/zones.py
ZONE_BBOX: dict[str, tuple[float, float, float, float]] = {**CHACO_ZONE, **TRAIN_ZONES, **EVAL_ZONES}

# las únicas 3 zonas del proyecto con algún polígono de desmonte (ver docs/v2/paper_metodologia.md §5.8);
# default de --zone cuando no se pasa ninguna.
DEFAULT_ZONES = ["chaco_santiago_frontier", "yungas", "periurbano_cordoba"]


def inspect_desmonte(rar: Path) -> None:
    "Esquema y distribuciones del shapefile completo, sin leer geometría (~3 s tras la primera extracción)."
    import pyogrio

    shp_path = str(G.ensure_shapefile_extracted(rar))
    info = pyogrio.read_info(shp_path)
    print(f"features: {info['features']:,}  geometry: {info['geometry_type']}  crs: {info['crs']}")
    print(f"bounds: {info['total_bounds']}")
    print(f"campos: {list(info['fields'])}")

    df = pyogrio.read_dataframe(shp_path, columns=G.DESM_COLS, read_geometry=False)
    print(f"\nFECHA_DESM: {df['FECHA_DESM'].nunique()} valores distintos, "
          f"min={df['FECHA_DESM'].min()} max={df['FECHA_DESM'].max()}")
    print(df["FECHA_DESM"].value_counts().sort_index().to_string())
    print(f"\nPROVINCIA:\n{df['PROVINCIA'].value_counts().to_string()}")
    print(f"\nSUPERF_ha:\n{df['SUPERF_ha'].describe().to_string()}")


def load_coords(zone: str, data_dir: Path) -> pd.DataFrame:
    path = P.latlon_file(zone, data_dir)
    if not path.exists():
        sys.exit(f"falta {rel(path)}")
    return pd.read_csv(path)


def centroid_window_positive(gdf: gpd.GeoDataFrame, coords_df: pd.DataFrame) -> pd.Series:
    """Método alternativo, solo para `--check-methods`: True si el CENTRO del píxel cae dentro
    de algún polígono con época en la ventana 2001-2022 (point-in-polygon, sin fracción)."""
    ventana = gdf[(gdf["epoca"] >= 2001) & (gdf["epoca"] <= 2022)]
    ids = coords_df["ID"].to_numpy()
    if ventana.empty:
        return pd.Series(False, index=ids)
    points = shapely.points(coords_df["longitude"].to_numpy(), coords_df["latitude"].to_numpy())
    tree = shapely.STRtree(ventana.geometry.to_numpy())
    point_idx, _ = tree.query(points, predicate="within")
    positive = np.zeros(len(ids), dtype=bool)
    positive[np.unique(point_idx)] = True
    return pd.Series(positive, index=ids)


def build_summary(coords_df: pd.DataFrame, agg: pd.DataFrame, mask: np.ndarray, grid: G.ZoneGrid) -> pd.DataFrame:
    "Resumen por píxel -- una fila por cada ID de coords_df (left join, sin huecos del lado del consumidor)."
    prev = agg.loc[agg["epoca"] == -1].set_index("ID")["frac"]
    post = agg.loc[agg["epoca"] == 9999].set_index("ID")["frac"]
    ventana = agg[(agg["epoca"] >= 2001) & (agg["epoca"] <= 2022)]

    frac_ventana = ventana.groupby("ID")["frac"].sum().clip(upper=1.0)
    # año dominante entre TODAS las épocas que tocan el píxel (empate -> la más antigua);
    # para -1/9999 no es "el año", es la bandera de línea de base/posterior -- ver docstring del módulo.
    dom = (
        agg.sort_values(["ID", "frac", "epoca"], ascending=[True, False, True])
        .groupby("ID", as_index=True)
        .first()[["epoca", "frac"]]
        .rename(columns={"epoca": "anio_dom", "frac": "frac_dom"})
    )
    n_epocas = agg.groupby("ID")["epoca"].nunique()
    n_pol = agg.groupby("ID")["n_pol"].sum()

    lat, lon = coords_df["latitude"].to_numpy(), coords_df["longitude"].to_numpy()
    i_id, j_id = G.grid_index(lat, lon)
    ii, jj = i_id - grid.i0, j_id - grid.j0
    relevado = mask[ii, jj]

    ids = coords_df["ID"]
    out = pd.DataFrame({
        "ID": ids,
        "frac_previo": ids.map(prev).fillna(0.0).to_numpy(),
        "frac_ventana": ids.map(frac_ventana).fillna(0.0).to_numpy(),
        "frac_post": ids.map(post).fillna(0.0).to_numpy(),
        "anio_dom": ids.map(dom["anio_dom"]).to_numpy(),
        "frac_dom": ids.map(dom["frac_dom"]).fillna(0.0).to_numpy(),
        "n_epocas": ids.map(n_epocas).fillna(0).astype(int).to_numpy(),
        "n_pol": ids.map(n_pol).fillna(0).astype(int).to_numpy(),
        "relevado": relevado,
    })
    return out


def check_totals(
    zone: str, gdf: gpd.GeoDataFrame, grid: G.ZoneGrid, coords_df: pd.DataFrame,
    agg: pd.DataFrame, poly_long: pd.DataFrame,
) -> dict[str, bool]:
    "Los chequeos de sanidad de la Fase 2 del preprocesamiento. Imprime y devuelve banderas ok/alerta."
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    flags: dict[str, bool] = {}
    id_to_i = pd.Series(
        G.grid_index(coords_df["latitude"].to_numpy(), coords_df["longitude"].to_numpy())[0],
        index=coords_df["ID"].to_numpy(),
    )

    print("  [1/5] conservación de área por época (Σ frac·área_píxel vs. área geodésica de la unión):")
    if not grid.complete:
        # La grilla tiene huecos (p. ej. yungas: solo quedaron las secuencias con
        # transición + una submuestra al 15% de las constantes, ver Fase 1 del plan).
        # `agg`/`frac` solo puede sumar área sobre los píxeles que SÍ existen en el
        # dataset -- cualquier parte de un polígono que caiga sobre un píxel
        # descartado desaparece de esa suma, mientras que `geod` mide el polígono
        # completo tal cual está en el shapefile. La brecha es sistemáticamente mayor
        # en las épocas con más peso de trayectorias constantes (p. ej. la línea de
        # base <=2000: si ya estaba desmontado antes de 2000 y no cambió más, su
        # trayectoria 2000-2022 es constante -- justo la clase que más se descartó).
        # No es comparable 1:1, así que se reporta informativo, sin marcar ALERTA.
        print("  [1/5] conservación de área por época: NO APLICA 1:1 -- grilla con huecos")
        for epoca, g in agg.groupby("epoca"):
            area_ha = float((pd.Series(G.pixel_area_ha(id_to_i.loc[g["ID"]].to_numpy())) * g["frac"].to_numpy()).sum())
            sub = gdf[gdf["epoca"] == epoca]
            if sub.empty:
                continue
            area_m2, _ = geod.geometry_area_perimeter(sub.geometry.union_all())
            area_geod_ha = abs(area_m2) / 10_000.0
            cobertura = area_ha / area_geod_ha if area_geod_ha else float("nan")
            print(f"        época {epoca:>5}: frac={area_ha:>12,.1f} ha  geod={area_geod_ha:>12,.1f} ha  "
                  f"cobertura={cobertura:.1%} (no es un error -- ver nota arriba)")
    else:
        area_ok = True
        for epoca, g in agg.groupby("epoca"):
            area_ha = float((pd.Series(G.pixel_area_ha(id_to_i.loc[g["ID"]].to_numpy())) * g["frac"].to_numpy()).sum())
            sub = gdf[gdf["epoca"] == epoca]
            if sub.empty or area_ha == 0:
                continue
            area_m2, _ = geod.geometry_area_perimeter(sub.geometry.union_all())
            area_geod_ha = abs(area_m2) / 10_000.0
            rel = abs(area_ha - area_geod_ha) / area_geod_ha if area_geod_ha else float("nan")
            # tolerancia 1,5%, no 0,5%: con unos pocos cientos/miles de polígonos por
            # época, el ruido de digitalización/redondeo de GEOS ronda ese orden y no
            # tiene una dirección sistemática (se verificó a mano: unas épocas dan
            # frac>geod, otras al revés) -- 0,5% resultó demasiado exigente en la
            # corrida real sobre chaco_santiago_frontier.
            ok = rel < 0.015
            area_ok &= ok
            print(f"        época {epoca:>5}: frac={area_ha:>12,.1f} ha  geod={area_geod_ha:>12,.1f} ha  "
                  f"diff={rel:.2%}  {'OK' if ok else 'ALERTA'}")
        flags["conservacion_area"] = area_ok

    print("  [2/5] frac <= 1 tras el recorte:")
    masa_total = float(agg["frac_raw"].sum())
    masa_recortada = float((agg["frac_raw"] - agg["frac"]).clip(lower=0).sum())
    pct = masa_recortada / masa_total if masa_total else 0.0
    flags["solapes_bajo_1pct"] = pct < 0.01
    print(f"        masa recortada por solapes: {masa_recortada:.2f} / {masa_total:.2f} ({pct:.2%})  "
          f"{'OK' if flags['solapes_bajo_1pct'] else 'ALERTA -- considerar --dissolve-epoch'}")

    print("  [3/5] alineación de grilla:")
    lat, lon = coords_df["latitude"].to_numpy(), coords_df["longitude"].to_numpy()
    i_chk, j_chk = G.grid_index(lat, lon)
    err_lat = float(np.abs((90.0 - (i_chk + 0.5) / 360.0) - lat).max())
    err_lon = float(np.abs((-180.0 + (j_chk + 0.5) / 360.0) - lon).max())
    # tolerancia 1e-5, no 1e-6: el netCDF fuente (ESA CCI) guarda lat/lon en float32 --
    # a esta magnitud (~25-70) su precisión relativa (~1,2e-7) ya da un error absoluto
    # de unos pocos 1e-6, antes de sumarle el redondeo a 6 decimales del CSV. 1e-6 era
    # más ajustado que la precisión real de la fuente (visto en la corrida real: hasta
    # 5,56e-6 en yungas, sin que la grilla esté mal alineada).
    flags["grilla_alineada"] = err_lat < 1e-5 and err_lon < 1e-5
    print(f"        error máx lat={err_lat:.2e} lon={err_lon:.2e}  {'OK' if flags['grilla_alineada'] else 'ALERTA'}")
    if grid.complete:
        flags["grilla_completa_cierra"] = grid.n_i * grid.n_j == len(coords_df)
        print(f"        grilla completa {grid.n_i}×{grid.n_j}={grid.n_i * grid.n_j:,} vs "
              f"{len(coords_df):,} filas  {'OK' if flags['grilla_completa_cierra'] else 'ALERTA'}")
    else:
        n_huecos = int((grid.ids == -1).sum())
        print(f"        grilla con huecos: {n_huecos:,} celdas vacías de {grid.n_i * grid.n_j:,} "
              f"(esperado en zonas submuestreadas, p. ej. yungas)")

    print("  [4/5] conteos por época y polígonos que tocan la grilla:")
    for epoca, n in sorted(gdf["epoca"].value_counts().items()):
        print(f"        época {epoca:>5}: {n:,} polígonos en el recorte de la zona")
    n_tocan = int(poly_long["poly_id"].nunique()) if not poly_long.empty else 0
    print(f"        polígonos que tocan al menos un píxel de la grilla: {n_tocan:,} / {len(gdf):,}")

    print("  [5/5] cobertura de la salida:")
    huerfanos = set(agg["ID"].unique()) - set(coords_df["ID"].unique())
    flags["sin_ids_huerfanos"] = not huerfanos
    print(f"        IDs en la salida ausentes de lat_long_df_{zone}.zip: {len(huerfanos)}  "
          f"{'OK' if flags['sin_ids_huerfanos'] else 'ALERTA'}")

    n_ok = sum(flags.values())
    print(f"  chequeos: {n_ok}/{len(flags)} OK")
    return flags


def run_zone(zone: str, args: argparse.Namespace) -> None:
    print(f"\n=== {zone} ===")
    if zone not in ZONE_BBOX:
        sys.exit(f"zona desconocida: {zone!r} (no está en land2vec.zones)")

    coords_df = load_coords(zone, args.data_dir)
    print(f"  grilla: {len(coords_df):,} píxeles en lat_long_df_{zone}.zip")
    grid = G.ZoneGrid.from_coords(coords_df, zone)
    print(f"  ventana de índices: {grid.n_i}×{grid.n_j}  {'completa' if grid.complete else 'con huecos'}")

    print(f"  leyendo shapefile (bbox de la zona)...")
    gdf = G.read_desmonte(args.rar, bbox=ZONE_BBOX[zone])
    print(f"  {len(gdf):,} polígonos en el bbox")

    gdf, clean_stats = G.clean_geometries(gdf)
    if clean_stats["n_invalidas_reparadas"]:
        print(f"  geometrías inválidas reparadas: {clean_stats['n_invalidas_reparadas']:,}")
    if clean_stats["n_descartadas"]:
        print(f"  geometrías descartadas (vacías tras reparar): {clean_stats['n_descartadas']:,}")

    print(f"  cruzando polígono<->píxel (fracción areal, {len(gdf):,} polígonos)...")
    poly_long = G.pixel_poly_fractions(gdf, grid, min_frac=args.min_frac)
    agg = G.pixel_fractions(gdf, grid, min_frac=args.min_frac, dissolve=args.dissolve_epoch)
    print(f"  {len(poly_long):,} filas (polígono, píxel) -- {len(agg):,} filas (píxel, época) agregadas")

    print(f"  máscara de área relevada (cell_deg={args.cell_deg}, dilate={args.dilate})...")
    mask = G.surveyed_mask(gdf, grid, cell_deg=args.cell_deg, dilate=args.dilate)
    print(f"  {int(mask.sum()):,} / {mask.size:,} celdas de la grilla dentro del área relevada "
          f"({mask.sum() / mask.size:.1%})")

    summary = build_summary(coords_df, agg, mask, grid)

    if args.check_methods:
        centroid = centroid_window_positive(gdf, coords_df)
        areal_pos = summary.set_index("ID")["frac_ventana"] >= 0.5
        cen = centroid.reindex(areal_pos.index).fillna(False)
        acuerdo = (areal_pos.to_numpy() == cen.to_numpy()).mean()
        print(f"  [check-methods] acuerdo areal(frac>=0.5) vs. centroide: {acuerdo:.2%} "
              f"(esperado >95%, desacuerdo concentrado en polígonos <2 px)")

    epoca_out = agg.rename(columns={"epoca": "anio"})[["ID", "anio", "frac", "n_pol"]]
    epoca_path = P.desmonte_epoca_file(zone, args.data_dir)
    epoca_out.to_csv(epoca_path, index=False, compression="zip")
    print(f"  guardado: {rel(epoca_path)} ({len(epoca_out):,} filas)")

    summary_path = P.desmonte_px_file(zone, args.data_dir)
    summary.to_csv(summary_path, index=False, compression="zip")
    print(f"  guardado: {rel(summary_path)} ({len(summary):,} filas)")

    poly_out = poly_long.rename(columns={"epoca": "anio"})[["poly_id", "ID", "anio", "frac"]].copy()
    # SUPERF_ha por poly_id -- necesaria en scripts/validacion/eval_desmonte.py para reconvertir `frac`
    # (fracción del PÍXEL) a `peso` (fracción del POLÍGONO) en la métrica de detección por
    # polígono; gdf.index == poly_id acá porque clean_geometries() ya reseteó el índice.
    poly_out["SUPERF_ha"] = poly_out["poly_id"].map(gdf["SUPERF_ha"])
    poly_path = P.desmonte_poly_file(zone, args.data_dir)
    poly_out.to_csv(poly_path, index=False, compression="zip")
    print(f"  guardado: {rel(poly_path)} ({len(poly_out):,} filas)")

    print("  verificación:")
    check_totals(zone, gdf, grid, coords_df, agg, poly_long)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inspect", action="store_true",
                         help="solo imprime esquema/distribuciones del shapefile completo y termina")
    parser.add_argument("--zone", action="append", default=None,
                         help="repetible; default: las 3 zonas con polígonos (chaco_santiago_frontier, "
                              "yungas, periurbano_cordoba)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--rar", type=Path, default=RAR_PATH)
    parser.add_argument("--method", choices=["areal", "centroid", "raster"], default="areal")
    parser.add_argument("--min-frac", type=float, default=1e-4)
    parser.add_argument("--cell-deg", type=float, default=0.1, help="tamaño de celda de surveyed_mask")
    parser.add_argument("--dilate", type=int, default=1, help="iteraciones de dilatación de surveyed_mask")
    parser.add_argument("--dissolve-epoch", action="store_true",
                         help="une los polígonos de cada época antes del cruce (más preciso ante "
                              "solapes, más caro) -- ver el diagnóstico del check [2/5]")
    parser.add_argument("--check-methods", action="store_true",
                         help="además, compara el método areal contra centroide (chequeo [6] del plan)")
    args = parser.parse_args()

    if args.inspect:
        inspect_desmonte(args.rar)
        return

    if args.method == "raster":
        sys.exit(
            "--method raster no está implementado todavía (opcional, requiere rasterio, no "
            "instalado en este entorno) -- usá --method areal (default) o --method centroid."
        )

    if not args.rar.exists():
        sys.exit(f"falta {args.rar}")

    zones = args.zone or DEFAULT_ZONES
    for zone in zones:
        run_zone(zone, args)


if __name__ == "__main__":
    main()
