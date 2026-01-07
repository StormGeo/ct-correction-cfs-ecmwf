"""Samet geostatistical interpolation pipeline."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt
from sklearn.neighbors import BallTree

LOGGER = logging.getLogger("geostats.samet")
DEFAULT_DEM_PATH = "/airflow/models/ct-near/data/etopo_0p01.nc"
_DEFAULT_BUFFER_KM = 12.5
_DEFAULT_VARIAGRAM_MODEL = "spherical"
_DEFAULT_TILE_SIZE = 512
_DEFAULT_N_CLOSEST = 24

for env_var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(env_var, "1")

try:  # pragma: no cover - optional dependency
    from global_land_mask import globe  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    globe = None

try:
    from pykrige.ok import OrdinaryKriging
except Exception as exc:  # pragma: no cover - pykrige optional at runtime
    OrdinaryKriging = None
    _PYKRIGE_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when dependency available
    _PYKRIGE_IMPORT_ERROR = None


def _ensure_lat_lon_names(ds: xr.Dataset, lat_name: str, lon_name: str) -> xr.Dataset:
    if lat_name not in ds.coords or lon_name not in ds.coords:
        raise KeyError(
            f"Latitude/longitude coordinates '{lat_name}'/'{lon_name}' not available."
        )
    return ds


def _detect_lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    lat_candidates = ["lat", "latitude", "y", "LAT", "Latitude"]
    lon_candidates = ["lon", "longitude", "x", "LON", "Longitude"]

    lat_name = next((name for name in lat_candidates if name in ds.coords), None)
    lon_name = next((name for name in lon_candidates if name in ds.coords), None)

    if lat_name is None or lon_name is None:
        raise KeyError(f"Unable to detect latitude/longitude coordinates in {list(ds.coords)}")

    return lat_name, lon_name


def _make_target_grid(ds: xr.Dataset, res_km: float, lat_name: str, lon_name: str) -> xr.Dataset:
    deg_per_km = 1.0 / 111.0
    step = res_km * deg_per_km

    lat = ds[lat_name].values
    lon = ds[lon_name].values

    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    

    lat_target = np.arange(
        np.floor(lat_min / step) * step,
        np.ceil(lat_max / step) * step + 1e-9,
        step,
    )
    lon_target = np.arange(
        np.floor(lon_min / step) * step,
        np.ceil(lon_max / step) * step + 1e-9,
        step,
    )

    return xr.Dataset({"lat": ("lat", lat_target), "lon": ("lon", lon_target)})


def _regrid_to(
    ds: xr.Dataset,
    varname: str,
    res_km: float,
    lat_name: str,
    lon_name: str,
) -> xr.Dataset:
    renamed = ds.rename({lat_name: "lat", lon_name: "lon"})

    if renamed.lat.values[0] > renamed.lat.values[-1]:
        renamed = renamed.sortby("lat")
    if renamed.lon.values[0] > renamed.lon.values[-1]:
        renamed = renamed.sortby("lon")

    target = _make_target_grid(renamed, res_km, "lat", "lon")

    out = (
        renamed[varname]
        .interp(lat=target.lat, lon=target.lon, method="linear")
        .to_dataset(name=varname)
    )
    return out


def _coerce_latlon_df(
    df: pd.DataFrame,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
) -> pd.DataFrame:
    try:  # pragma: no cover - geopandas optional
        import geopandas as gpd  # type: ignore

        if isinstance(df, gpd.GeoDataFrame) and getattr(df, "geometry", None) is not None:
            geo = df.dropna(subset=["geometry"]).copy()
            geom_type = getattr(geo.geometry, "geom_type", pd.Series()).astype(str).str.upper()
            if (geom_type == "POINT").any() or (geom_type == "POINT").all():
                return pd.DataFrame({"lat": geo.geometry.y.values, "lon": geo.geometry.x.values})
    except Exception:
        pass

    df_in = df.copy()
    if lat_col and lon_col:
        if lat_col not in df_in.columns or lon_col not in df_in.columns:
            raise KeyError(
                f"Columns lat_col={lat_col}/lon_col={lon_col} not found. Available: {list(df_in.columns)}"
            )
        return (
            df_in[[lat_col, lon_col]]
            .rename(columns={lat_col: "lat", lon_col: "lon"})
            .dropna()
        )

    candidate_lats = ["lat", "latitude", "Latitude", "LAT", "Lat", "y", "Y"]
    candidate_lons = ["lon", "longitude", "Longitude", "LON", "Lon", "x", "X"]
    lat_name = next((col for col in candidate_lats if col in df_in.columns), None)
    lon_name = next((col for col in candidate_lons if col in df_in.columns), None)
    if lat_name and lon_name:
        return (
            df_in[[lat_name, lon_name]]
            .rename(columns={lat_name: "lat", lon_name: "lon"})
            .dropna()
        )

    numeric_cols = [col for col in df_in.columns if pd.api.types.is_numeric_dtype(df_in[col])]
    if len(numeric_cols) >= 2:
        guesses: Iterable[Tuple[str, str]] = []
        for col in numeric_cols:
            series = df_in[col].astype(float)
            if series.between(-90, 90).mean() > 0.9:
                guesses = [*guesses, ("lat", col)]
            elif series.between(-180, 180).mean() > 0.9:
                guesses = [*guesses, ("lon", col)]
        lat_guess = next((col for kind, col in guesses if kind == "lat"), None)
        lon_guess = next((col for kind, col in guesses if kind == "lon"), None)
        if lat_guess and lon_guess:
            return (
                df_in[[lat_guess, lon_guess]]
                .rename(columns={lat_guess: "lat", lon_guess: "lon"})
                .dropna()
            )

    raise KeyError(f"Could not identify latitude/longitude columns. Available: {list(df_in.columns)}")


def _build_station_mask(
    grid_lat: xr.DataArray,
    grid_lon: xr.DataArray,
    stations_df: pd.DataFrame,
    radius_km: float,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
) -> xr.DataArray:
    stations = _coerce_latlon_df(stations_df, lat_col=lat_col, lon_col=lon_col)
    if stations.empty:
        raise ValueError("No valid station coordinates found.")

    lats = grid_lat.values
    lons = grid_lon.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    points_rad = np.radians(points)
    stations_rad = np.radians(stations[["lat", "lon"]].values)

    tree = BallTree(stations_rad, metric="haversine")
    dist_rad, _ = tree.query(points_rad, k=1, return_distance=True)

    earth_radius_km = 6371.0
    dist_km = dist_rad.ravel() * earth_radius_km
    mask = (dist_km <= radius_km).reshape(lat_grid.shape)

    return xr.DataArray(mask, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))


def _sample_dem(
    orog: xr.Dataset,
    df_points: pd.DataFrame,
    candidates: Tuple[str, ...] = ("z", "elevation", "ETOPO", "orog", "topography", "height"),
) -> np.ndarray:
    orog_var = next((name for name in candidates if name in orog.data_vars), None)
    if orog_var is None:
        raise ValueError("Elevation variable not found in DEM dataset.")

    dataset = orog
    if "lat" not in dataset.coords:
        for name in ["latitude", "y", "LAT", "Latitude"]:
            if name in dataset.coords:
                dataset = dataset.rename({name: "lat"})
                break
    if "lon" not in dataset.coords:
        for name in ["longitude", "x", "LON", "Longitude"]:
            if name in dataset.coords:
                dataset = dataset.rename({name: "lon"})
                break

    if dataset["lat"][0] > dataset["lat"][-1]:
        dataset = dataset.sortby("lat")
    if dataset["lon"][0] > dataset["lon"][-1]:
        dataset = dataset.sortby("lon")

    lon_min = float(dataset.lon.min())
    point_lon = df_points["lon"].to_numpy().astype(float)
    if lon_min >= 0.0 and point_lon.min() < 0.0:
        point_lon = point_lon % 360.0
    if lon_min < 0.0 and point_lon.max() > 180.0:
        point_lon = ((point_lon + 180.0) % 360.0) - 180.0

    lats_query = xr.DataArray(df_points["lat"].to_numpy().astype(float), dims="points")
    lons_query = xr.DataArray(point_lon, dims="points")
    sampled = dataset[orog_var].sel(lat=lats_query, lon=lons_query, method="nearest")

    values = sampled.to_numpy().astype("float32")
    values[values < 0] = 0.0
    return values


def _adjust_to_msl(
    df_points: pd.DataFrame,
    temp_col: str = "value",
    alt_col: str = "altitude",
    lapse_rate: float = 0.0065,
) -> pd.Series:
    return df_points[temp_col] + df_points[alt_col] * lapse_rate


def _deduplicate_points(df_points: pd.DataFrame, tol_m: float = 15.0) -> pd.DataFrame:
    if df_points.empty:
        return df_points

    df = df_points.copy()
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)

    tol_deg = tol_m / 111_000.0

    df["_ilat"] = np.round(df["lat"] / tol_deg).astype(np.int64)
    df["_ilon"] = np.round(df["lon"] / tol_deg).astype(np.int64)

    priority = {"station": 0, "model": 1}
    rows = []
    for _, group in df.groupby(["_ilat", "_ilon"], sort=False):
        group = group.sort_values("source", key=lambda series: series.map(priority))
        top_source = group.iloc[0]["source"]
        selected = group[group["source"] == top_source]

        lat = float(selected["lat"].mean())
        lon = float(selected["lon"].mean())
        value = float(selected["value"].mean())

        rows.append({"lat": lat, "lon": lon, "value": value, "source": top_source})

    out = pd.DataFrame(rows, columns=["lat", "lon", "value", "source"])
    out.attrs["dedup_tol_m"] = tol_m
    out.attrs["n_before"] = int(len(df_points))
    out.attrs["n_after"] = int(len(out))
    out.attrs["n_removed"] = int(len(df_points) - len(out))
    return out


def get_mask_far_from_continent(
    dataset: xr.DataArray | xr.Dataset,
    *,
    lat_name: str = None,
    lon_name: str = None,
    buffer_km: float = _DEFAULT_BUFFER_KM,
    return_mask_only: bool = True,
) -> xr.DataArray | xr.Dataset:
    if globe is None:
        raise ImportError("global_land_mask is required to compute coast-aware mask")

    obj = dataset
    lat_candidates = ["lat", "latitude", "y", "LAT", "Latitude"]
    lon_candidates = ["lon", "longitude", "x", "LON", "Longitude"]

    if lat_name is None:
        lat_name = next((name for name in lat_candidates if name in obj.coords), None)
    if lon_name is None:
        lon_name = next((name for name in lon_candidates if name in obj.coords), None)
    if lat_name is None or lon_name is None:
        raise KeyError(f"Could not locate lat/lon coordinates in {list(obj.coords)}")

    lats = obj.coords[lat_name].values
    lons = obj.coords[lon_name].values

    if lats[0] > lats[-1]:
        obj = obj.sortby(lat_name)
        lats = obj.coords[lat_name].values
    if lons[0] > lons[-1]:
        obj = obj.sortby(lon_name)
        lons = obj.coords[lon_name].values

    lon2d, lat2d = np.meshgrid(lons, lats)
    is_land = globe.is_land(lat2d, lon2d)

    fn_lat = -3.85
    fn_lon = -32.42
    fn_i = np.abs(lat2d[:, 0] - fn_lat).argmin()
    fn_j = np.abs(lon2d[0, :] - fn_lon).argmin()
    is_land[fn_i, fn_j] = True

    if len(lats) > 1:
        dy_km = abs(lats[1] - lats[0]) * 111.32
    else:
        dy_km = 111.32
    if len(lons) > 1:
        lat_mid = float((lats[0] + lats[-1]) / 2.0)
        dx_km = abs(lons[1] - lons[0]) * 111.32 * np.cos(np.deg2rad(lat_mid))
        dx_km = max(dx_km, 1e-6)
    else:
        dx_km = 111.32 * np.cos(np.deg2rad(float(lats.mean()))) if lats.size else 111.32

    dist_km = distance_transform_edt(~is_land, sampling=(dy_km, dx_km))
    mask_far = dist_km > buffer_km

    mask_da = xr.DataArray(mask_far, dims=[lat_name, lon_name], coords={lat_name: lats, lon_name: lons})

    if return_mask_only:
        return mask_da

    if isinstance(obj, xr.DataArray):
        return obj.where(~mask_da)

    dataset_out = obj.copy()
    for name in dataset_out.data_vars:
        data_array = dataset_out[name]
        dims = set(data_array.dims)
        if lat_name in dims and lon_name in dims:
            dataset_out[name] = data_array.where(~mask_da)
    return dataset_out



def reajustar_para_superficie(
    ds_nmm: xr.Dataset,
    var: str,
    dem_path: str,
    ddeg: float = 0.125,
    lapse_rate: float = 0.0065,
    out_nc: Optional[str] = None,
) -> str:
    orog = xr.open_dataset(dem_path)
    try:
        if "lat" not in orog.coords:
            for candidate in ["latitude", "y", "LAT", "Latitude"]:
                if candidate in orog.coords:
                    orog = orog.rename({candidate: "lat"})
                    break
        if "lon" not in orog.coords:
            for candidate in ["longitude", "x", "LON", "Longitude"]:
                if candidate in orog.coords:
                    orog = orog.rename({candidate: "lon"})
                    break
        if orog["lat"][0] > orog["lat"][-1]:
            orog = orog.sortby("lat")
        if orog["lon"][0] > orog["lon"][-1]:
            orog = orog.sortby("lon")

        def _make_regular_grid(ds_like: xr.Dataset, step: float = 0.125) -> xr.Dataset:
            lat = ds_like["lat"].values
            lon = ds_like["lon"].values
            lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
            lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
            lat0 = np.floor(lat_min / step) * step
            lat1 = np.ceil(lat_max / step) * step
            lon0 = np.floor(lon_min / step) * step
            lon1 = np.ceil(lon_max / step) * step
            lat_t = np.arange(lat0, lat1 + 1e-9, step)
            lon_t = np.arange(lon0, lon1 + 1e-9, step)
            return xr.Dataset({"lat": ("lat", lat_t), "lon": ("lon", lon_t)})

        da_var = ds_nmm[var]
        reference_time: Optional[pd.Timestamp] = None
        if "time" in da_var.dims:
            t_coord = ds_nmm["time"]
            reference_time = pd.to_datetime(t_coord.values[0])
            da_var = da_var.isel(time=0)

        if "lat" not in da_var.dims or "lon" not in da_var.dims:
            da_var = da_var.rename({da_var.dims[-2]: "lat", da_var.dims[-1]: "lon"})

        #target = _make_regular_grid(da_var, step=ddeg)
        lat_target = np.arange(-53.0, 22.875 + 1e-6, ddeg)
        lon_target = np.arange(-88.7, -30.45 + 1e-6, ddeg)
        target = xr.Dataset({"lat": ("lat", lat_target), "lon": ("lon", lon_target)})

        orog_var = next(
            name for name in ["z", "elevation", "ETOPO", "orog", "topography", "height"] if name in orog.data_vars
        )

        dem_da = orog[orog_var]
        if "lat" not in dem_da.dims or "lon" not in dem_da.dims:
            dem_da = dem_da.rename({dem_da.dims[-2]: "lat", dem_da.dims[-1]: "lon"})
        dem_da = dem_da.interp(lat=target["lat"], lon=target["lon"], method="linear")
        dem_da = xr.where(dem_da < 0, 0, dem_da)

        t_msl = da_var.interp(lat=target["lat"], lon=target["lon"], method="linear")
        t_surface = (t_msl - lapse_rate * dem_da).astype("float32")
        t_surface.name = var
        t_surface.attrs.update(
            {
                "long_name": f"{var} (surface, de-adjusted from MSL)",
                "units": da_var.attrs.get("units", "degC"),
            }
        )

        ds_surface = xr.Dataset({var: t_surface}, coords={"lat": target["lat"], "lon": target["lon"]})
        ds_surface.attrs.update(
            {
                "title": f"{var} surface ({ddeg}°) from MSL kriging + DEM de-adjustment",
                "history": "xarray.interp bilinear + lapse-rate de-adjustment",
            }
        )

        ds_surface = ds_surface.rename({"lat": "latitude", "lon": "longitude"})
        if ds_surface.latitude.values[0] > ds_surface.latitude.values[-1]:
            ds_surface = ds_surface.sortby("latitude")
        if ds_surface.longitude.values[0] > ds_surface.longitude.values[-1]:
            ds_surface = ds_surface.sortby("longitude")

        ds_surface["latitude"].attrs.update(
            {"units": "degrees_north", "long_name": "latitude", "axis": "Y"}
        )
        ds_surface["longitude"].attrs.update(
            {"units": "degrees_east", "long_name": "longitude", "axis": "X"}
        )

        if reference_time is None or pd.isna(reference_time):
            reference_time = pd.Timestamp.utcnow().tz_localize(None).normalize()
        ref0 = pd.Timestamp(reference_time.date())
        hours = np.array([(reference_time - ref0) / pd.Timedelta(hours=1)], dtype="float64")
        ds_surface = ds_surface.expand_dims(time=1)
        ds_surface = ds_surface.assign_coords(time=("time", hours))
        ds_surface["time"].attrs.update(
            {
                "standard_name": "time",
                "long_name": "time",
                "units": f"hours since {ref0.strftime('%Y-%m-%d %H:%M:%S')}",
                "calendar": "standard",
            }
        )

        stdname_map = {"2m_air_temperature": "air_temperature"}
        var_da = ds_surface[var].astype("float64")
        if var in stdname_map:
            var_da.attrs["standard_name"] = stdname_map[var]
            var_da.attrs["height"] = 2.0
        ds_surface[var] = var_da

        ds_surface = ds_surface.transpose("time", "latitude", "longitude")

        ds_surface.attrs.update(
            {
                "Conventions": "CF-1.8",
                "institution": "StormGeo",
                "source": ds_nmm.attrs.get("source", "unknown"),
                "history": (
                    ds_surface.attrs.get("history", "") + " | saved CF lon/lat for ncview"
                ).strip(),
            }
        )

        out_nc_final = out_nc or str(Path("geostats_output") / f"{var}_surface_{ddeg}deg.nc")
        Path(out_nc_final).parent.mkdir(parents=True, exist_ok=True)

        encoding = {
            var: {"zlib": True, "complevel": 4, "dtype": "float64"},
            "time": {"zlib": True, "complevel": 4, "dtype": "float64"},
            "latitude": {"zlib": True, "complevel": 4, "dtype": "float64"},
            "longitude": {"zlib": True, "complevel": 4, "dtype": "float64"},
        }

        ds_surface = ds_surface.sel(
            longitude=slice(-88.7, -30.45),
            latitude=slice(-53., 22.875)
        )

        level_value = 1000.0
        ds_surface = ds_surface.expand_dims(level=[level_value])
        ds_surface["level"].attrs.update({
            "units": "hPa",
            "long_name": "pressure level"
        })

        ds_surface = ds_surface.transpose("time", "level", "latitude", "longitude")

        da = ds_surface[var].isel(level=0)
        ds_surface[var] = da
        ds_surface[var].attrs.update({
            "long_name": f"{var} surface",
            "units": "K"
        })

        ds_surface.attrs.clear()
        ds_surface.attrs.update({
            "Conventions": "CF-1.4",
            "Metadata_Conventions": "Unidata Dataset Discovery v1.0",
            "HISTORY": "Created by Ismael ct_near_NEW..."
        })

        ds_surface.to_netcdf(out_nc_final, encoding=encoding)


    finally:
        orog.close()

    return out_nc_final


def _krige_streaming_parallel(
    ds_masked: xr.Dataset,
    varname: str,
    ok: Any,
    out_nc: str,
    *,
    ds_time_source: Optional[xr.Dataset] = None,
    tile_lat: int = _DEFAULT_TILE_SIZE,
    tile_lon: int = _DEFAULT_TILE_SIZE,
    fill_only_holes: bool = False,
    write_variance: bool = False,
    n_closest_points: Optional[int] = None,
    backend: Optional[str] = None,
    complevel: int = 1,
    dtype: str = "f4",
    max_workers: Optional[int] = None,
) -> None:
    ds_masked = ds_masked.sortby("lat").sortby("lon")
    data_array = ds_masked[varname]
    has_time = "time" in data_array.dims

    time_values = None
    if ds_time_source is not None and "time" in ds_time_source.coords:
        coord = ds_time_source["time"]
        if coord.size >= 1:
            time_values = np.array(coord.values).ravel()[:1]

    write_time = has_time or (time_values is not None)

    if has_time:
        base = data_array.isel(time=0).values.astype("float32", copy=False)
    else:
        base = data_array.values.astype("float32", copy=False)

    lats = ds_masked["lat"].values
    lons = ds_masked["lon"].values
    nlat, nlon = len(lats), len(lons)
    holes = np.isnan(base) if fill_only_holes else None

    def _compute_tile(i0: int, i1: int, j0: int, j1: int):
        y_block = lats[i0:i1]
        x_block = lons[j0:j1]

        if fill_only_holes and not np.any(holes[i0:i1, j0:j1]):
            return i0, i1, j0, j1, base[i0:i1, j0:j1], None

        exec_backend = backend
        if n_closest_points is not None and (exec_backend is None or exec_backend == "vectorized"):
            exec_backend = "loop"

        kwargs: Dict[str, Any] = {}
        if n_closest_points is not None:
            kwargs["n_closest_points"] = n_closest_points
        if exec_backend is not None:
            kwargs["backend"] = exec_backend

        try:
            z_block, var_block = ok.execute("grid", x_block, y_block, **kwargs)
        except ValueError as err:
            if (
                ("moving window" in str(err).lower() or "n_closest_points" in str(err).lower())
                and kwargs.get("backend") != "loop"
            ):
                kwargs["backend"] = "loop"
                z_block, var_block = ok.execute("grid", x_block, y_block, **kwargs)
            else:
                raise
        except TypeError:
            z_block, var_block = ok.execute("grid", x_block, y_block)

        z_block = z_block.astype("float32", copy=False)

        if fill_only_holes:
            original = base[i0:i1, j0:j1]
            out_block = np.where(np.isnan(original), z_block, original).astype("float32", copy=False)
        else:
            out_block = z_block

        variance_block = var_block.astype("float32", copy=False) if write_variance else None
        return i0, i1, j0, j1, out_block, variance_block

    with Dataset(out_nc, "w") as nc_file:
        if write_time:
            nc_file.createDimension("time", 1)
        nc_file.createDimension("lat", nlat)
        nc_file.createDimension("lon", nlon)

        vlat = nc_file.createVariable("lat", "f4", ("lat",))
        vlon = nc_file.createVariable("lon", "f4", ("lon",))
        vlat[:] = lats.astype("float32")
        vlat.units = "degrees_north"
        vlon[:] = lons.astype("float32")
        vlon.units = "degrees_east"

        if write_time:
            vtime = nc_file.createVariable("time", "i8", ("time",))
            if time_values is None:
                np_time = np.array(ds_masked["time"].values).ravel()[:1]
            else:
                np_time = time_values
            timestamp = pd.Timestamp(np_time[0])
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            epoch = pd.Timestamp("1970-01-01T00:00:00Z")
            vtime.units = "seconds since 1970-01-01 00:00:00 UTC"
            vtime.calendar = "standard"
            vtime[:] = int((timestamp - epoch).total_seconds())

        clat = max(1, min(tile_lat, nlat))
        clon = max(1, min(tile_lon, nlon))

        if write_time:
            v = nc_file.createVariable(
                varname,
                dtype,
                ("time", "lat", "lon"),
                zlib=True,
                complevel=complevel,
                chunksizes=(1, clat, clon),
                fill_value=np.float32(np.nan),
            )
            vvar = None
            if write_variance:
                vvar = nc_file.createVariable(
                    f"{varname}_kriging_variance",
                    dtype,
                    ("time", "lat", "lon"),
                    zlib=True,
                    complevel=complevel,
                    chunksizes=(1, clat, clon),
                    fill_value=np.float32(np.nan),
                )
        else:
            v = nc_file.createVariable(
                varname,
                dtype,
                ("lat", "lon"),
                zlib=True,
                complevel=complevel,
                chunksizes=(clat, clon),
                fill_value=np.float32(np.nan),
            )
            vvar = None
            if write_variance:
                vvar = nc_file.createVariable(
                    f"{varname}_kriging_variance",
                    dtype,
                    ("lat", "lon"),
                    zlib=True,
                    complevel=complevel,
                    chunksizes=(clat, clon),
                    fill_value=np.float32(np.nan),
                )

        v.long_name = f"{varname} (kriged)"
        v.units = getattr(ds_masked[varname], "attrs", {}).get("units", "")
        if write_variance and vvar is not None:
            vvar.long_name = "Kriging variance"
            vvar.units = v.units

        tiles = [
            (i0, min(i0 + tile_lat, nlat), j0, min(j0 + tile_lon, nlon))
            for i0 in range(0, nlat, tile_lat)
            for j0 in range(0, nlon, tile_lon)
        ]

        workers = max_workers if max_workers is not None else (os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_compute_tile, *tile) for tile in tiles]
            for future in as_completed(futures):
                i0, i1, j0, j1, out_block, variance_block = future.result()
                if write_time:
                    v[0, i0:i1, j0:j1] = out_block
                    if write_variance and vvar is not None and variance_block is not None:
                        vvar[0, i0:i1, j0:j1] = variance_block
                else:
                    v[i0:i1, j0:j1] = out_block
                    if write_variance and vvar is not None and variance_block is not None:
                        vvar[i0:i1, j0:j1] = variance_block

        nc_file.title = f"Kriged {varname} (samet pipeline)"
        nc_file.history = "Created by PyKrige with tile streaming"


def _parse_reference_time(args, df: pd.DataFrame) -> Optional[pd.Timestamp]:
    datetime_value = getattr(args, "datetime_str", None)
    if datetime_value:
        for fmt in ("%Y%m%d%H", "%Y%m%d%H%M", None):
            try:
                return pd.to_datetime(datetime_value, format=fmt) if fmt else pd.to_datetime(datetime_value)
            except (TypeError, ValueError):
                continue
        raise ValueError(f"Could not parse datetime string '{datetime_value}'")

    if "datetime" in df.columns and not df["datetime"].empty:
        return pd.to_datetime(df["datetime"].iloc[0])

    return None


def _prepare_station_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for source, target in [
        ("latitude", "lat"),
        ("lat", "lat"),
        ("longitude", "lon"),
        ("lon", "lon"),
        ("data", "value"),
        ("value", "value"),
        ("altitude", "altitude"),
        ("height", "altitude"),
    ]:
        if source in df.columns:
            rename_map[source] = target

    stations = df.rename(columns=rename_map)
    required = {"lat", "lon", "value"}
    if not required.issubset(stations.columns):
        raise KeyError("Stations dataframe must provide latitude/longitude/value columns.")

    columns = ["lat", "lon", "value"]
    if "altitude" in stations.columns:
        columns.append("altitude")
    stations = stations[columns].copy()
    if "altitude" not in stations.columns:
        stations["altitude"] = 0.0

    stations = stations.dropna(subset=["lat", "lon", "value"])
    for column in ["lat", "lon", "value", "altitude"]:
        stations[column] = stations[column].astype(float)
    stations["source"] = "station"
    return stations


def _collect_model_points(da: xr.DataArray) -> pd.DataFrame:
    df_model = da.to_dataframe(name="value").dropna().reset_index()
    if "time" in df_model.columns:
        df_model = df_model.drop(columns=["time"])
    df_model = df_model.rename(columns={"lat": "lat", "lon": "lon"})
    df_model["source"] = "model"
    return df_model


def _samet_krige(
    ds: xr.Dataset,
    ds_time_source: xr.Dataset,
    varname: str,
    stations_df: pd.DataFrame,
    *,
    lat_name: str,
    lon_name: str,
    res_km: float,
    mask_radius_km: float,
    dem_path: Optional[str],
    use_temp_msl: bool,
    lapse_rate: float,
    out_path: str,
    tile_lat: int,
    tile_lon: int,
    n_closest_points: Optional[int],
    max_workers: Optional[int],
    variogram_model: str,
    points_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    ds_checked = _ensure_lat_lon_names(ds, lat_name, lon_name)
    ds_rg = _regrid_to(ds_checked, varname, res_km=res_km, lat_name=lat_name, lon_name=lon_name)

    if not stations_df.empty:
        mask = _build_station_mask(
            ds_rg["lat"], ds_rg["lon"], stations_df, radius_km=mask_radius_km
        )
        da_masked = ds_rg[varname].where(~mask)
    else:
        da_masked = ds_rg[varname]

    df_model = _collect_model_points(da_masked)
    model_points = df_model[["lat", "lon", "value"]].copy()
    if "altitude" not in model_points.columns:
        model_points["altitude"] = 0.0
    model_points["source"] = "model"

    stations = stations_df[["lat", "lon", "value", "altitude", "source"]]
    df_points = pd.concat([stations, model_points], ignore_index=True)

    df_points = _deduplicate_points(df_points, tol_m=15.0)
    LOGGER.debug(
        "[samet] dedup tol≈%sm | before=%s after=%s removed=%s",
        df_points.attrs.get("dedup_tol_m"),
        df_points.attrs.get("n_before"),
        df_points.attrs.get("n_after"),
        df_points.attrs.get("n_removed"),
    )

    values = df_points["value"].values
    if dem_path is not None:
        with xr.open_dataset(dem_path) as dem:
            altitudes = _sample_dem(dem, df_points)
        df_points["altitude"] = altitudes
        if use_temp_msl:
            values = _adjust_to_msl(df_points, lapse_rate=lapse_rate).values

    if points_csv_path:
        csv_path = Path(points_csv_path)
        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            export_df = df_points.copy()
            if "altitude" not in export_df.columns:
                export_df["altitude"] = 0.0
            export_df["value_for_kriging"] = np.asarray(values, dtype=float)
            export_df.to_csv(csv_path, index=False)
            LOGGER.info("[samet] saved interpolation dataframe to %s", csv_path)
        except Exception as exc:
            LOGGER.warning("[samet] failed to export interpolation dataframe: %s", exc)

    ok = OrdinaryKriging(
        df_points["lon"].values,
        df_points["lat"].values,
        values,
        variogram_model=variogram_model,
        enable_plotting=False,
        verbose=False,
    )

    ds_masked = da_masked.to_dataset(name=varname).sortby("lat").sortby("lon")

    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)

    _krige_streaming_parallel(
        ds_masked=ds_masked,
        varname=varname,
        ok=ok,
        out_nc=str(out_path_obj),
        ds_time_source=ds_time_source,
        tile_lat=tile_lat,
        tile_lon=tile_lon,
        fill_only_holes=False,
        write_variance=False,
        n_closest_points=n_closest_points,
        backend=None,
        complevel=1,
        dtype="f4",
        max_workers=max_workers,
    )

    return {
        "out_nc": str(out_path_obj),
        "grid": {
            "lat_size": int(ds_masked.dims["lat"]),
            "lon_size": int(ds_masked.dims["lon"]),
            "res_km": res_km,
        },
    }


def run_samet_interpolation(df: pd.DataFrame, args) -> Dict[str, Any]:
    if OrdinaryKriging is None:
        raise ImportError(f"PyKrige is not available: {_PYKRIGE_IMPORT_ERROR}")

    if not args.merge_path:
        raise ValueError("samet interpolation requires a model netCDF file provided via --merge")

    if not os.path.exists(args.merge_path):
        raise FileNotFoundError(f"Model file not found: {args.merge_path}")

    varname = args.vname
    LOGGER.info("[samet] opening model dataset %s", args.merge_path)
    ds_model = xr.open_dataset(args.merge_path)

    if "glo" in args.merge_path:
        LOGGER.info("[samet] applying spatial subset for global file")
        ds_model = ds_model.sel(
            longitude=slice(-88.7, -30.45),
            latitude=slice(-53., 22.875)
        )


    LOGGER.info("[samet] model dataset ready: %s", ds_model)

    lat_name, lon_name = _detect_lat_lon_names(ds_model)

    reference_time = _parse_reference_time(args, df)
    if reference_time is not None and "time" in ds_model.coords:
        try:
            ds_model = ds_model.sel(time=reference_time)
        except Exception:
            ds_model = ds_model.sel(time=reference_time, method="nearest")
    ds_time_source = ds_model

    if globe is not None:
        try:
            ds_model = get_mask_far_from_continent(
                ds_model, lat_name=lat_name, lon_name=lon_name, buffer_km=_DEFAULT_BUFFER_KM, return_mask_only=False
            )
        except Exception as exc:  # pragma: no cover - mask failures shouldn't break pipeline
            LOGGER.warning("[samet] unable to apply coast-aware mask: %s", exc)

    stations = _prepare_station_dataframe(df)

    res_km = float(args.res) * 111.32
    mask_radius_km = float(args.interpolation_radius)
    dem_path = DEFAULT_DEM_PATH if args.use_topo else None
    if dem_path is not None and not Path(dem_path).exists():
        LOGGER.warning("[samet] DEM file not found at %s, altitude adjustment disabled", dem_path)
        dem_path = None
    use_temp_msl = bool(args.use_topo)
    lapse_rate = 0.0065

    variogram_model = getattr(args, "method_function", None) or _DEFAULT_VARIAGRAM_MODEL

    final_output_path = Path(args.out_path)
    if final_output_path.suffix:
        kriging_out_path = final_output_path.with_name(
            f"{final_output_path.stem}_msl{final_output_path.suffix}"
        )
    else:
        kriging_out_path = final_output_path.with_suffix(".msl.nc")
    points_csv_path = final_output_path.with_name(f"{final_output_path.stem}_points.csv")

    result = _samet_krige(
        ds_model,
        ds_time_source,
        varname,
        stations,
        lat_name=lat_name,
        lon_name=lon_name,
        res_km=27,
        mask_radius_km=mask_radius_km,
        dem_path=dem_path,
        use_temp_msl=use_temp_msl,
        lapse_rate=lapse_rate,
        out_path=str(kriging_out_path),
        tile_lat=_DEFAULT_TILE_SIZE,
        tile_lon=_DEFAULT_TILE_SIZE,
        n_closest_points=_DEFAULT_N_CLOSEST,
        max_workers=os.cpu_count() or 4,
        variogram_model=variogram_model,
        points_csv_path=str(points_csv_path),
    )

    dem_for_surface = dem_path
    surface_path = str(final_output_path)
    if dem_for_surface is None:
        LOGGER.info("[samet] surface adjustment disabled (DEM not provided)")

    ds_nmm = xr.open_dataset(result["out_nc"])
    try:
        ds_surface = ds_nmm
        if globe is not None:
            try:
                ds_surface = get_mask_far_from_continent(
                    ds_surface,
                    lat_name="lat",
                    lon_name="lon",
                    buffer_km=50.0,
                    return_mask_only=False,
                )
            except Exception as exc:  # pragma: no cover - soft failure
                LOGGER.warning("[samet] unable to mask surface dataset: %s", exc)

        if dem_for_surface is not None:
            surface_path = reajustar_para_superficie(
                ds_nmm=ds_surface,
                var=varname,
                dem_path=dem_for_surface,
                ddeg=float(args.res),
                lapse_rate=lapse_rate,
                out_nc=str(final_output_path),
            )
        else:
            ds_surface.to_netcdf(str(final_output_path))
            surface_path = str(final_output_path)
    finally:
        ds_nmm.close()

    result["kriging_out_nc"] = result["out_nc"]
    result["out_nc"] = surface_path
    if points_csv_path is not None:
        result["points_csv"] = str(points_csv_path)

    LOGGER.info("[samet] output written to %s", result["out_nc"])
    for dataset in (ds_model, ds_time_source):
        try:
            dataset.close()
        except Exception:
            pass
    return result


__all__ = ["run_samet_interpolation", "get_mask_far_from_continent", "reajustar_para_superficie"]
