"""Merge interpolation pipeline (Barnes 1-pass with optional background field)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger("merge")


DEFAULT_RADIUS_KM = 250.0
N_MIN = 4
USE_SECTORS = True
N_SECTORS = 4
MAX_PER_SECTOR = 2
MAX_NEIGHBORS = 32
EPS_FAR_KM = 20.0

ALLOW_EXTRAP = True
EXTRAP_K = 3
EXTRAP_DECAY_KM = 400.0
N_MIN_EXTRAP = 1

DEFAULT_RES_DEG = 0.125
GRID_PAD_DEG = 0.25

FILL_NAN_BY_NN = True

BATCH_QUERY_RADIUS = True
BATCH_CHUNK_SIZE = 50000

EARTH_RADIUS_KM = 6371.0


try:  # pragma: no cover - scikit-learn is optional at runtime
    from sklearn.neighbors import BallTree, KDTree

    _HAS_SK = True
except Exception:  # pragma: no cover
    BallTree = KDTree = None
    _HAS_SK = False


def _snap(val: float, step: float) -> float:
    return np.floor(val / step) * step


def make_grid_from_points(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    res_deg: float = DEFAULT_RES_DEG,
    pad_deg: float = GRID_PAD_DEG,
) -> Tuple[np.ndarray, np.ndarray]:
    if lat.size == 0:
        raise ValueError("merge interpolation received no valid points")

    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)

    lat_min = _snap(lat_min - pad_deg, res_deg)
    lat_max = _snap(lat_max + pad_deg, res_deg)
    lon_min = _snap(lon_min - pad_deg, res_deg)
    lon_max = _snap(lon_max + pad_deg, res_deg)

    lats = np.arange(lat_min, lat_max + 1e-9, res_deg, dtype=np.float64)
    lons = np.arange(lon_min, lon_max + 1e-9, res_deg, dtype=np.float64)
    return lats, lons


def _bearing_sector(lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    if not USE_SECTORS or N_SECTORS <= 1:
        return np.zeros_like(lats, int)

    lat0r = np.deg2rad(lat0)
    dlat = np.deg2rad(lats - lat0)
    dlon = np.deg2rad(lons - lon0)
    ang = (np.arctan2(dlon * np.cos(lat0r), dlat) + 2 * np.pi) % (2 * np.pi)
    return np.floor(ang / (2 * np.pi / N_SECTORS)).astype(int)


def _sector_cap(lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray, dkm: np.ndarray) -> np.ndarray:
    if not USE_SECTORS or N_SECTORS <= 1:
        return np.argsort(dkm)[:MAX_NEIGHBORS]

    keep: list[int] = []
    sec = _bearing_sector(lat0, lon0, lats, lons)
    for sector in range(N_SECTORS):
        idx = np.where(sec == sector)[0]
        if idx.size == 0:
            continue
        order = np.argsort(dkm[idx])
        keep.extend(idx[order[:MAX_PER_SECTOR]])

    keep_arr = np.array(keep, int)
    if keep_arr.size > MAX_NEIGHBORS:
        keep_arr = keep_arr[np.argsort(dkm[keep_arr])[:MAX_NEIGHBORS]]
    return keep_arr


def _build_balltree(lat_pts: np.ndarray, lon_pts: np.ndarray) -> Optional[BallTree]:
    if not _HAS_SK or lat_pts.size == 0:
        return None
    return BallTree(np.radians(np.c_[lat_pts, lon_pts]), metric="haversine")


def _neighbors_batched(
    tree: Optional[BallTree],
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    radius_km: float,
) -> Tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Versão otimizada:
    - Constrói a malha apenas uma vez
    - Converte todos os pontos para radianos antes de fatiar em batches
    """
    if tree is None:
        return [], []

    ny, nx = grid_lats.size, grid_lons.size
    yy, xx = np.meshgrid(grid_lats, grid_lons, indexing="ij")
    # (Ngrid, 2)
    points = np.c_[yy.ravel(), xx.ravel()]
    # Converte tudo de uma vez para radianos
    points_rad = np.radians(points)
    R_rad = radius_km / EARTH_RADIUS_KM

    n_points = points_rad.shape[0]
    inds_all: list[np.ndarray] = []
    dists_all: list[np.ndarray] = []

    for start in range(0, n_points, BATCH_CHUNK_SIZE):
        end = min(start + BATCH_CHUNK_SIZE, n_points)
        chunk = points_rad[start:end, :]
        inds, dists = tree.query_radius(
            chunk,
            r=R_rad,
            return_distance=True,
            sort_results=True,
        )
        inds_all.extend(inds)
        dists_all.extend(dists)

    return inds_all, dists_all


def _nearest_knn_batch(
    tree: Optional[BallTree],
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    k: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if tree is None or len(lat_arr) == 0:
        return None, None
    dist, ind = tree.query(np.radians(np.c_[lat_arr, lon_arr]), k=k)
    return ind, dist * EARTH_RADIUS_KM


def fill_nans_by_nearest(grid: np.ndarray) -> np.ndarray:
    g = grid.astype("float64", copy=True)
    mask = np.isfinite(g)
    if mask.all():
        return g

    if not _HAS_SK:
        yy, xx = np.indices(g.shape)
        yv, xv = yy[mask], xx[mask]
        vals = g[mask]
        yn, xn = yy[~mask], xx[~mask]
        d2 = (yn[:, None] - yv[None, :]) ** 2 + (xn[:, None] - xv[None, :]) ** 2
        g[~mask] = vals[np.argmin(d2, axis=1)]
        return g

    yy, xx = np.nonzero(mask)
    yn, xn = np.nonzero(~mask)
    tree = KDTree(np.c_[yy, xx])
    _, ind = tree.query(np.c_[yn, xn], k=1, return_distance=True)
    g[yn, xn] = g[(yy[ind[:, 0]], xx[ind[:, 0]])]
    return g


def _haversine_km_vec(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Versão vetorizada e reutilizável do haversine para o caminho fallback.
    """
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * (2.0 * np.arcsin(np.sqrt(a)))


def barnes_1pass_from_df(
    df_all: pd.DataFrame,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    *,
    radius_km: float,
) -> np.ndarray:
    """
    Versão reescrita/otimizada do Barnes 1-pass:

    - Usa BallTree + query_radius em batch (se disponível).
    - Evita np.ndindex e reduz overhead de loops.
    - Mantém lógica de setores, extrapolação e pesos.
    """

    # Filtra pontos válidos
    pts = df_all.dropna(subset=["lat", "lon", "value"])

    lat_pts = pts["lat"].to_numpy(np.float64)
    lon_pts = pts["lon"].to_numpy(np.float64)
    val_pts = pts["value"].to_numpy(np.float64)
    if "w0" in pts.columns:
        w0_pts = pts["w0"].to_numpy(np.float64)
    else:
        w0_pts = np.ones_like(val_pts, dtype=np.float64)

    ny, nx = grid_lats.size, grid_lons.size
    out = np.full((ny, nx), np.nan, dtype=np.float64)

    # Se não há pontos, devolve grade toda NaN
    if lat_pts.size == 0:
        return out

    tree = _build_balltree(lat_pts, lon_pts)

    # Caminho rápido: BallTree + consultas em batch
    if tree is not None and BATCH_QUERY_RADIUS:
        inds_all, dists_all = _neighbors_batched(tree, grid_lats, grid_lons, radius_km)

        # Se por algum motivo não vier nada, caímos no fallback depois
        if inds_all:
            empty_flat: list[int] = []

            radius_km32 = np.float64(radius_km)
            eps_far32 = np.float64(EPS_FAR_KM)
            n_grid = ny * nx

            for k in range(n_grid):
                idx = inds_all[k]
                if idx.size == 0:
                    empty_flat.append(k)
                    continue

                # distâncias em km (BallTree/haversine → radianos)
                dkm = (dists_all[k].astype(np.float64) * EARTH_RADIUS_KM).astype(
                    np.float64
                )

                # Índices de linha/coluna na grade
                i = k // nx
                j = k % nx

                lat0 = float(grid_lats[i])
                lon0 = float(grid_lons[j])

                # Limita por setores para evitar muitos vizinhos em uma mesma direção
                sel = _sector_cap(lat0, lon0, lat_pts[idx], lon_pts[idx], dkm)
                if sel.size == 0:
                    empty_flat.append(k)
                    continue

                idx_sel = idx[sel]
                dkm_sel = dkm[sel]

                if idx_sel.size < N_MIN:
                    empty_flat.append(k)
                    continue

                d_eff = np.maximum(dkm_sel, eps_far32)
                # pesos gaussianos
                w = w0_pts[idx_sel] * np.exp(-((d_eff / radius_km32) ** 2))
                den = w.sum(dtype=np.float64)
                if den > 0.0:
                    out[i, j] = np.dot(w, val_pts[idx_sel]) / den

            # Extrapolação opcional para pontos vazios
            if ALLOW_EXTRAP and empty_flat:
                empty_flat_arr = np.array(empty_flat, dtype=int)
                yi = empty_flat_arr // nx
                xi = empty_flat_arr % nx

                # K vizinhos mais próximos (não limitados por radius_km)
                k_extrap = min(EXTRAP_K, lat_pts.size)
                ind, dkm = _nearest_knn_batch(
                    tree,
                    grid_lats[yi].astype(np.float64),
                    grid_lons[xi].astype(np.float64),
                    k=k_extrap,
                )

                if ind is not None and dkm is not None:
                    # pesos decaindo com EXTRAP_DECAY_KM
                    w = (
                        w0_pts[ind]
                        * np.exp(-((dkm.astype(np.float64) / EXTRAP_DECAY_KM) ** 2))
                    ).astype(np.float64)
                    den = w.sum(axis=1)
                    ok = den > 0.0
                    if np.any(ok):
                        out[yi[ok], xi[ok]] = (
                            (w[ok] * val_pts[ind[ok]]).sum(axis=1) / den[ok]
                        )

            return out

    # ------------------------------------------------------------------
    # Fallback: sem BallTree (ou BATCH_QUERY_RADIUS=False) → versão
    # haversine com loops, mas com helper vetorizado.
    # ------------------------------------------------------------------
    for i, y0 in enumerate(grid_lats):
        for j, x0 in enumerate(grid_lons):
            dkm_all = _haversine_km_vec(
                float(y0),
                float(x0),
                lat_pts,
                lon_pts,
            )
            mask = dkm_all <= radius_km
            if np.count_nonzero(mask) < N_MIN:
                continue

            cand_lat = lat_pts[mask]
            cand_lon = lon_pts[mask]
            dkm = dkm_all[mask]

            sel = _sector_cap(float(y0), float(x0), cand_lat, cand_lon, dkm)
            if sel.size == 0:
                continue

            idx = np.where(mask)[0][sel]
            dkm_sel = dkm[sel]

            d_eff = np.maximum(dkm_sel, EPS_FAR_KM)
            w = w0_pts[idx] * np.exp(-((d_eff / radius_km) ** 2))
            den = w.sum()
            if den > 0.0:
                out[i, j] = np.nansum(w * val_pts[idx]) / den

    return out


def build_df_all(
    stations_df: pd.DataFrame,
    sat_nc: Optional[str],
    target_time: Optional[np.datetime64],
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    val_col: str = "data",
    bg_weight: float = 0.40,
) -> pd.DataFrame:
    df = stations_df.copy()

    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])

    if target_time is not None and "datetime" in df.columns:
        dt_series = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        target_ts = pd.to_datetime(target_time, utc=True)
        df = df[dt_series == target_ts]

    df_st = pd.DataFrame(
        {
            "lat": pd.to_numeric(df.get(lat_col, np.nan), errors="coerce").astype("float64"),
            "lon": pd.to_numeric(df.get(lon_col, np.nan), errors="coerce").astype("float64"),
            "value": pd.to_numeric(df.get(val_col, np.nan), errors="coerce").astype("float64"),
            "w0": np.float64(1.0),
            "source": "station",
        }
    ).dropna(subset=["lat", "lon", "value"]).reset_index(drop=True)

    if not sat_nc or not os.path.exists(sat_nc):
        if sat_nc and not os.path.exists(sat_nc):
            LOGGER.warning("[merge] background file not found at %s, skipping", sat_nc)
        return df_st[["lat", "lon", "value", "w0", "source"]]

    ds = xr.open_dataset(sat_nc)
    try:
        var_candidates = list(ds.data_vars)
        if not var_candidates:
            raise ValueError("background dataset has no variables")
        var = "precipitation" if "precipitation" in ds.data_vars else var_candidates[0]
        da = ds[var]

        if "latitude" in da.coords and "lat" not in da.coords:
            da = da.rename({"latitude": "lat"})
        if "longitude" in da.coords and "lon" not in da.coords:
            da = da.rename({"longitude": "lon"})

        if "time" in da.dims and target_time is not None:
            times = da["time"].values
            if times.size:
                idx = int(np.argmin(np.abs(times - np.datetime64(target_time))))
                da = da.isel(time=idx)

        if "lat" not in da.dims or "lon" not in da.dims:
            raise ValueError("background dataset must include lat/lon coordinates")

        non_spatial_dims = [dim for dim in da.dims if dim not in ("lat", "lon")]
        if non_spatial_dims:
            da = da.transpose(*(non_spatial_dims + ["lat", "lon"]))
            da = da.isel({dim: 0 for dim in non_spatial_dims})
        else:
            da = da.transpose("lat", "lon")

        sat_field = np.asarray(da.astype("float64").values, np.float64)
        sat_lats = np.asarray(da["lat"].values, np.float64)
        sat_lons = np.asarray(da["lon"].values, np.float64)

        if sat_field.ndim != 2:
            raise ValueError("background field must be 2-D after selection")

        i = np.searchsorted(sat_lats, df_st["lat"].to_numpy(np.float64)).clip(0, sat_lats.size - 1)
        j = np.searchsorted(sat_lons, df_st["lon"].to_numpy(np.float64)).clip(0, sat_lons.size - 1)
        mask = np.zeros_like(sat_field, dtype=bool)
        for di in range(-2, 3):
            ii = (i + di).clip(0, sat_field.shape[0] - 1)
            for dj in range(-2, 3):
                jj = (j + dj).clip(0, sat_field.shape[1] - 1)
                mask[ii, jj] = True
        sat_field[mask] = np.nan

        ii, jj = np.where(~np.isnan(sat_field))
        df_bg = pd.DataFrame(
            {
                "lat": sat_lats[ii].astype("float64"),
                "lon": sat_lons[jj].astype("float64"),
                "value": sat_field[ii, jj].astype("float64"),
                "w0": np.float64(bg_weight),
                "source": "background",
            }
        )

        return pd.concat([df_st, df_bg], ignore_index=True)[["lat", "lon", "value", "w0", "source"]]
    finally:
        ds.close()


def run_from_df_all(
    df_all: pd.DataFrame,
    *,
    var_name: str,
    out_nc: Optional[str],
    reference_time: Optional[np.datetime64],
    res_deg: float = DEFAULT_RES_DEG,
    radius_km: float = DEFAULT_RADIUS_KM,
) -> Dict[str, object]:
    lat_min, lat_max = -53.0, 22.875
    lon_min, lon_max = -88.7, -30.45

    lats_out = np.arange(lat_min, lat_max + 1e-9, res_deg, dtype=np.float64)
    lons_out = np.arange(lon_min, lon_max + 1e-9, res_deg, dtype=np.float64)


    grid = barnes_1pass_from_df(df_all, lats_out, lons_out, radius_km=radius_km)
    if FILL_NAN_BY_NN:
        grid = fill_nans_by_nearest(grid)

    result: Dict[str, object] = {
        "grid": grid,
        "latitude": lats_out,
        "longitude": lons_out,
        "radius_km": radius_km,
    }

    if out_nc:
        Path(out_nc).parent.mkdir(parents=True, exist_ok=True)
        timestamp = reference_time if reference_time is not None else np.datetime64("now")
        level_value = 1000.0
        

        ds_out = xr.Dataset(
            {
                var_name: (
                    ("time", "latitude", "longitude"),
                    grid[np.newaxis, ...].astype("float64"),
                )
            },
            coords={
                "time": np.array([timestamp], dtype="datetime64[ns]"),
                "latitude": lats_out.astype("float64"),
                "longitude": lons_out.astype("float64"),
            },
        )

        ds_out[var_name].attrs.update({
            "long_name": f"{var_name}",
        })
        ds_out["latitude"].attrs.update({
            "units": "degrees_north",
            "long_name": "latitude",
            "axis": "Y",
        })
        ds_out["longitude"].attrs.update({
            "units": "degrees_east",
            "long_name": "longitude",
            "axis": "X",
        })

        encoding = {
            var_name: {"zlib": True, "complevel": 4, "dtype": "float64"},
            "time": {"dtype": "float64"},
            "latitude": {"dtype": "float64"},
            "longitude": {"dtype": "float64"},
        }

        ds_out.to_netcdf(out_nc, encoding=encoding)
        ds_out.close()
        result["out_nc"] = out_nc

    return result


def _parse_reference_time(args, df: pd.DataFrame) -> Optional[np.datetime64]:
    dt_str = getattr(args, "datetime_str", None)
    ts: Optional[pd.Timestamp] = None

    if dt_str:
        for fmt in ("%Y%m%d%H", "%Y%m%d%H%M"):
            try:
                ts = pd.to_datetime(dt_str, format=fmt, utc=True)
                break
            except Exception:
                ts = None
        if ts is None:
            try:
                ts = pd.to_datetime(dt_str, utc=True)
            except Exception:
                LOGGER.warning("[merge] unable to parse datetime %s", dt_str)

    if ts is None and "datetime" in df.columns and not df["datetime"].empty:
        ts = pd.to_datetime(df["datetime"].iloc[0], utc=True, errors="coerce")

    if ts is None or pd.isna(ts):
        return None

    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
        ts = ts.tz_localize(None)

    return np.datetime64(ts)


def run_merge_interpolation(df: pd.DataFrame, args) -> Dict[str, object]:
    reference_time = _parse_reference_time(args, df)
    LOGGER.info("[merge] reference time resolved to %s", reference_time)

    sat_path = getattr(args, "merge_path", None)
    df_all = build_df_all(df, sat_path, reference_time)

    if df_all.empty:
        raise ValueError("merge interpolation received no valid stations or background points")

    LOGGER.info("[merge] using %s points for interpolation", len(df_all))

    radius_arg = getattr(args, "interpolation_radius", None)
    try:
        radius_km = float(radius_arg) if radius_arg is not None else DEFAULT_RADIUS_KM
    except Exception:
        LOGGER.warning("[merge] invalid interpolation radius %s, falling back to %.1f km", radius_arg, DEFAULT_RADIUS_KM)
        radius_km = DEFAULT_RADIUS_KM

    if radius_km <= 0:
        LOGGER.warning("[merge] non-positive radius %.3f km provided; using default %.1f km", radius_km, DEFAULT_RADIUS_KM)
        radius_km = DEFAULT_RADIUS_KM

    LOGGER.info("[merge] using %.1f km influence radius", radius_km)

    result = run_from_df_all(
        df_all,
        var_name=args.vname,
        out_nc=args.out_path,
        reference_time=reference_time,
        res_deg=float(args.res),
        radius_km=radius_km,
    )

    final_output = Path(args.out_path)
    points_csv = final_output.with_name(f"{final_output.stem}_points.csv") if final_output.suffix else final_output.with_suffix(".points.csv")
    try:
        points_csv.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(points_csv, index=False)
        result["points_csv"] = str(points_csv)
        LOGGER.info("[merge] saved interpolation dataframe to %s", points_csv)
    except Exception as exc:  # pragma: no cover - CSV export best-effort
        LOGGER.warning("[merge] unable to export interpolation dataframe: %s", exc)

    if "out_nc" not in result:
        result["out_nc"] = args.out_path

    LOGGER.info("[merge] output written to %s", result["out_nc"])
    return result


__all__ = ["run_merge_interpolation"]
