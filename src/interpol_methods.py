import logging as log
import os
import traceback
import datetime

import netCDF4 as nc
import numpy as np

import pytz as tz

import scipy.interpolate

from ncBuilder import ncBuilder
from ncBuilder import ncHelper

from scipy.signal import savgol_filter

from src.bias_spread import spread_correction
from src.commons import PROVIDER_LIST
from src.commons import ensure_interpol
from src.commons import load_config
from src.methods.barnes import apply_barnes
from src.methods.bilinear import apply_bilinear
from src.methods.cressman import apply_cressman
from src.methods.fill_nans import fill_nans_parallel
from src.methods.idw import apply_idw
from src.methods.kriging import apply_kriging
from src.methods.kriging import apply_universal_kriging
from src.methods.metpy import apply_metpy
from src.methods.rbf import apply_rbf
from src.methods.rbf import apply_tbs
from src.methods.simple import apply_cubic
from src.methods.simple import apply_linear
from src.methods.simple import apply_nearest
from src.methods.spatial_circle import apply_circles
from src.methods.spline import apply_spline

REINSERTION_ORDER = ['ogimet', 'redemet', 'inmet']


def fix_grid(data, lat_orig, lon_orig, lat, lon, method='cubic'):
    X, Y = np.meshgrid(lon_orig, lat_orig)
    XI, YI = np.meshgrid(lon, lat)
    new_grid = scipy.interpolate.griddata(
        (X.flatten(), Y.flatten()), data.flatten(), (XI, YI), method=method)
    return new_grid


def get_indexes(latvals, lonvals, lat0_rad, lon0_rad):
    minindex_x = np.abs(latvals[:] - lat0_rad).argmin()
    minindex_y = np.abs(lonvals[:] - lon0_rad).argmin()
    return minindex_x, minindex_y


def calculate_dew_point(temp, rh):
    alpha = np.divide(np.multiply(17.27, temp), np.add(
        237.7, temp)) + np.log(np.divide(rh, 100.0))
    dew = np.divide(np.multiply(237.7, alpha), np.subtract(17.27, alpha))
    return np.array(dew, dtype=np.float32)


def calculate_relative_humidity(temp, dew):
    ea = 6.108 * np.exp((17.27 * dew) / (dew + 273.3))
    es = 6.108 * np.exp((17.27 * temp) / (temp + 273.3))
    rh = (ea / es) * 100
    rh[rh > 100] = 99.9
    return rh


def level_fixer(data, topo, vname, topo_max=None):
    if vname in ['2m_air_temperature', '2m_dew_point']:
        return data + (6.5 * topo / 1000)
    elif vname in ['10m_wind', '10m_wind_direct']:
        if not topo_max:
            topo_max = np.nanmax(topo) // 4
        return data * np.power((topo + 10) / (topo_max + 11), 0.14)
    else:
        return data


def level_fixer_undo(data, topo, vname, topo_max=None):
    if vname in ['2m_air_temperature', '2m_dew_point',
                 '2m_dew_point_temperature']:
        return data - (6.5 * topo / 1000)

    elif vname in ['10m_wind', '10m_wind_direct']:
        if not topo_max:
            topo_max = np.nanmax(topo, axis=(0, 1)) // 4
        return data * np.power((topo_max + 11) / (topo + 10), 0.14)
    else:
        return data


def load_topo(args, lats, lons):
    if args.domain == [-53.0, 22.0, -88.7, -31.4] and args.res == 0.125:
        log.info('Using default topography')
        topo_file = nc.Dataset(
            '/airflow/models/ct-near/data/etopo_ct_near.nc', 'r')
        topo = topo_file.variables['z'][:]
        topo[topo < 0] = 0.
        topo_regrid = topo.copy()
        topo_file.close()

    elif args.domain == [-53.0, 22.0, -88.7, -31.4] and args.res == 0.15:
        log.info('Using default 0p15 topography')
        topo_file = nc.Dataset(
            '/airflow/models/ct-near/data/etopo_ct_near_0p15.nc', 'r')
        topo = topo_file.variables['z'][:]
        topo[topo < 0] = 0.
        topo_regrid = topo.copy()
        topo_file.close()

    else:
        log.info('Using HR topography')
        topo_file = nc.Dataset(
            '/airflow/models/ct-near/data/etopo_0p01.nc', 'r')
        topo = topo_file.variables['z'][:]
        tlats, tlons = ncHelper.get_lats_lons(topo_file)
        topo[topo < 0] = 0.
        topo_regrid = fix_grid(topo, tlats, tlons, lats, lons)
        topo_file.close()
    return topo_regrid


def merge_background(args, grid, lats, lons, topo_regrid):

    grid = level_fixer_undo(grid, topo_regrid, args.vname)

    configs = load_config(args.vname)
    merge_file = nc.Dataset(args.merge_path, 'r')
    mlat, mlon = ncHelper.get_lats_lons(merge_file)
    merge = merge_file.variables[args.vname][-1, ...]
    merge = fix_grid(merge, mlat, mlon, lats, lons)

    if np.any(np.isnan(grid)):
        mask = ~np.isnan(merge) | np.isnan(grid)
    else:
        mask = np.zeros_like(grid, dtype=bool)

    if configs['merge_rules']['upper_limit']:
        mask = mask | (grid > configs['merge_rules']['upper_limit'])
    if configs['merge_rules']['lower_limit']:
        mask = mask | (grid > configs['merge_rules']['lower_limit'])

    grid[mask] = merge[mask]
    grid = level_fixer(grid, topo_regrid, args.vname)

    return grid


def reinsert_points(grid, df, lats, lons, args):

    log.info(f"src.interpol_methods.reinsert_points >> starting point reinsertion")

    ilat = [np.argmin(abs(i - lats)) for i in df['latitude'].values]
    ilon = [np.argmin(abs(i - lons)) for i in df['longitude'].values]

    for ix, iy, val, prov in zip(ilat, ilon, df['data'].values, df[
                                 'station_provider']):

        if prov in REINSERTION_ORDER:
            continue

        if prov in PROVIDER_LIST:
            if args.vname in ['total_precipitation', '10m_wind_gust', '10m_wind_direct',
                              '10m_wind_speed', '10m_u_component_of_wind',
                              '10m_v_component_of_wind', '2m_air_temperature']:

                if val != 0:
                    if args.vname in ['total_precipitation']:
                        cond_insertion = [
                            (grid[ix + 1 if ix + 1 <
                                  grid.shape[0] else ix, iy] != 0),
                            (grid[ix - 1 if ix > 0 else ix, iy] != 0),
                            (grid[ix, iy + 1 if iy + 1 <
                                  grid.shape[1] else iy] != 0),
                            (grid[ix, iy - 1 if iy > 0 else iy] != 0)
                        ]

                        if np.abs(grid[ix, iy]) < np.abs(
                                val) and np.any(cond_insertion):
                            grid[ix, iy] = val

                    else:
                        grid[ix, iy] = val

            else:
                grid[ix, iy] = val

    for prov in REINSERTION_ORDER:
        df_prov = df[df['station_provider'] == prov]

        if df_prov.empty:
            log.warning(f"src.interpol_methods.reinsert_points >> not found any station to {prov} provider")
            continue

        data = df_prov['data'].values
        ilat = [np.argmin(abs(i - lats)) for i in df_prov['latitude'].values]
        ilon = [np.argmin(abs(i - lons)) for i in df_prov['longitude'].values]

        for ix, iy, val in zip(ilat, ilon, data):
            if args.vname in ['total_precipitation', '10m_wind_gust', '10m_wind_direct',
                              '10m_wind_speed', '10m_u_component_of_wind',
                              '10m_v_component_of_wind', '2m_air_temperature']:
                if val != 0:
                    grid[ix, iy] = val
            else:
                grid[ix, iy] = val

    return grid


@ensure_interpol(method_on_exception='cressman')
def apply_interpol(args, plats, plons,
                   palts, data, glats, glons, galts, interpol_method=None):
    '''
    plats, plons, glats, glons are in radians
    '''

    if interpol_method is None:
        interpol_method = args.interpol_method

    data = level_fixer(np.array(data), np.array(palts), args.vname)

    if interpol_method == 'cressman':
        apply_method = apply_cressman
    elif interpol_method == 'barnes':
        apply_method = apply_barnes
    elif interpol_method == 'idw':
        apply_method = apply_idw
    elif interpol_method == 'rbf':
        apply_method = apply_rbf
    elif interpol_method == 'spline':
        apply_method = apply_spline
    elif interpol_method == 'linear':
        apply_method = apply_linear
    elif interpol_method == 'cubic':
        apply_method = apply_cubic
    elif interpol_method == 'nearest':
        apply_method = apply_nearest
    elif interpol_method == 'metpy':
        apply_method = apply_metpy
    elif interpol_method == 'kriging':
        apply_method = apply_kriging
    elif interpol_method == 'ukriging':
        apply_method = apply_universal_kriging
    elif interpol_method == 'tbs':
        apply_method = apply_tbs
    else:
        apply_method = apply_circles

    grid = apply_method(
        plats,
        plons,
        palts,
        data,
        glats,
        glons,
        galts,
        args)

    return grid


def apply_composed(args, df, lats, lons, topo_regrid):
    grid1 = apply_interpol(args,
                           df.loc[df['altitude'] <= args.altitude_threshold][
                               'latitude'].values,
                           df.loc[df['altitude'] <= args.altitude_threshold][
                               'longitude'].values,
                           df.loc[df['altitude'] <= args.altitude_threshold][
                               'altitude'].values,
                           df.loc[
                               df['altitude'] <= args.altitude_threshold]['data'].values,
                           lats,
                           lons,
                           topo_regrid,
                           interpol_method=args.interpol_method1)

    grid2 = apply_interpol(args,
                           df.loc[df['altitude'] > args.altitude_threshold][
                               'latitude'].values,
                           df.loc[df['altitude'] > args.altitude_threshold][
                               'longitude'].values,
                           df.loc[df['altitude'] > args.altitude_threshold][
                               'altitude'].values,
                           df.loc[
                               df['altitude'] > args.altitude_threshold]['data'].values,
                           lats,
                           lons,
                           topo_regrid,
                           interpol_method=args.interpol_method2)

    grid = np.where(topo_regrid > args.altitude_threshold, grid2, grid1)
    return grid


def interpol_data(df, args, clear_sky_mask=None, use_topo=True):

    special_methods = {'samet', 'merge'}
    if args.composed_method and special_methods & {args.interpol_method1, args.interpol_method2}:
        raise ValueError('samet/merge interpolation is not available with composed_method enabled')

    if args.interpol_method == 'samet':
        from src.methods.samet import run_samet_interpolation

        log.info('src.interpol_methods.interpol_data >> dispatching to samet interpolation pipeline')
        run_samet_interpolation(df.copy(), args)
        return

    if args.interpol_method == 'merge':
        from src.methods.merge import run_merge_interpolation

        log.info('src.interpol_methods.interpol_data >> dispatching to merge interpolation pipeline')
        run_merge_interpolation(df.copy(), args)
        return

    log.info('src.interpol_methods.interpol_data >> starting interpolation')
    lats_deg = np.arange(args.domain[0], args.domain[1] + 1, float(args.res))
    lons_deg = np.arange(args.domain[2], args.domain[3] + 1, float(args.res))

    lats = np.radians(lats_deg)
    lons = np.radians(lons_deg)
    df['latitude'] = np.radians(np.array(df['latitude'].values, dtype=float))
    df['longitude'] = np.radians(np.array(df['longitude'].values, dtype=float))

    data = np.array(df['data'].values)

    topo_regrid = load_topo(args, lats_deg, lons_deg)

    if args.vname == '2m_air_temperature':

        if args.composed_method:
            grid = apply_composed(args, df, lats, lons, topo_regrid)

        else:

            grid = apply_interpol(args,
                                  df['latitude'].values,
                                  df['longitude'].values,
                                  df['altitude'].values,
                                  data,
                                  lats,
                                  lons,
                                  topo_regrid)

        if args.merge_path is not None:
            grid = merge_background(
                args, grid, lats_deg, lons_deg, topo_regrid)

        grid = fill_nans_parallel(
            grid, max_distance=(args.interpolation_radius * 2 / (112 * float(args.res))))

        if args.spread_correction:
            cgrid = spread_correction(args,
                                      np.array(
                                          df['latitude'].values, dtype=float),
                                      np.array(
                                          df['longitude'].values, dtype=float),
                                      data,
                                      lats,
                                      lons,
                                      grid,
                                      radius=1800)

        grid = savgol_filter(grid, 4, 2, axis=0)
        grid = savgol_filter(grid, 4, 2, axis=1)

        if args.spread_correction:
            if cgrid.T == grid.shape:
                grid = grid + cgrid.T
            else:
                grid = grid + cgrid

        grid = level_fixer_undo(grid, topo_regrid, args.vname)

        grid[grid >= 47] = np.nan
        grid[grid <= -18] = np.nan

    elif args.vname == '2m_relative_humidity':
        fake_temp = 22.4

        data = calculate_dew_point(np.full(len(df['data'].values), fake_temp),
                                   np.array(df['data'].values, dtype=float))

        data = level_fixer(data,
                           np.array(df['altitude'].values),
                           '2m_air_temperature')

        grid = apply_interpol(args,
                              df['latitude'].values,
                              df['longitude'].values,
                              df['altitude'].values,
                              data,
                              lats,
                              lons,
                              topo_regrid)

        grid = fill_nans_parallel(grid, max_distance=(
            args.interpolation_radius * 2 / (112 * float(args.res))))

        if args.spread_correction:
            cgrid = spread_correction(args,
                                      np.array(
                                          df['latitude'].values, dtype=float),
                                      np.array(
                                          df['longitude'].values, dtype=float),
                                      data,
                                      lats,
                                      lons,
                                      grid,
                                      radius=1800)

        grid = savgol_filter(grid, 4, 2, axis=0)
        grid = savgol_filter(grid, 4, 2, axis=1)

        if args.spread_correction:
            if cgrid.T == grid.shape:
                grid = grid + cgrid.T
            else:
                grid = grid + cgrid

        grid = level_fixer_undo(grid, topo_regrid, '2m_air_temperature')

        grid = calculate_relative_humidity(
            np.full(grid.shape, fake_temp), grid)

        # merging with relative humidity merge_path
        if args.merge_path is not None:
            grid = merge_background(
                args, grid, lats_deg, lons_deg, topo_regrid)

        grid[grid < 0] = np.nan
        grid[grid > 100] = np.nan

    elif args.vname == 'total_precipitation':

        if clear_sky_mask is not None:
            ds_cs = clear_sky_mask

        else:
            ds_cs = None

        data = np.array(df['data'].values, dtype=float)
        data = data + 1  # spread 0s
        grid = apply_interpol(args,
                              df['latitude'].values,
                              df['longitude'].values,
                              df['altitude'].values,
                              data,
                              lats,
                              lons,
                              topo_regrid)

        grid = grid - 1
        grid[grid < -0.01] = np.nan

        grid = savgol_filter(grid, 4, 2, axis=0)
        grid = savgol_filter(grid, 4, 2, axis=1)

        if not args.merge_path is None:

            log.info(f"src.interpol_methods >> loading merge data: {args.merge_path}")

            merge_file = nc.Dataset(args.merge_path, 'r')
            mlat, mlon = ncHelper.get_lats_lons(merge_file)
            merge = merge_file.variables[args.vname][...]

            if merge.shape[0] == 1:
                merge = merge[0, ...]
            else:
                mtimes = merge_file.variables['time']
                mtimes = nc.num2date(mtimes[:], mtimes.units)
                mtimes = [datetime.datetime.fromisoformat(i.isoformat()) for i in mtimes]
                mtimes = [tz.utc.localize(i) for i in mtimes]
                dtnow = df.sort_values(
                    'datetime').iloc[-1]['datetime'].to_pydatetime()

                step = np.argmin([abs((i - dtnow).total_seconds()) for i in mtimes])

                merge = merge[step, ...]

            lat_max, lat_min = mlat[-1], mlat[0]
            lon_max, lon_min = mlon[-1], mlon[0]

            # get index
            idlat_max = np.nanargmin(np.abs(lats_deg - lat_max))
            idlat_min = np.nanargmin(np.abs(lats_deg - lat_min))
            idlon_max = np.nanargmin(np.abs(lons_deg - lon_max))
            idlon_min = np.nanargmin(np.abs(lons_deg - lon_min))

            # merge to ct-near grid
            merge = fix_grid(
                merge,
                mlat,
                mlon,
                lats_deg,
                lons_deg,
                method='nearest')

            mask = (grid < 1) | (np.isnan(grid))

            # get max value in intersection
            grid = np.where((grid >= 1) & (merge >= 1),
                            np.nanmax(np.array([grid, merge]), axis=0), grid)

            merge[:idlat_min, :] = np.nan
            merge[idlat_max:, :] = np.nan
            merge[:, idlon_max:] = np.nan
            merge[:, :idlon_min] = np.nan

            # non_zero_mask = merge != 0
            # replace_mask = np.logical_and(mask, non_zero_mask)

            # apply merge
            grid[mask] = merge[mask]

        # apply clear sky mask
        if ds_cs is not None:

            log.info(f"src.interpol_methods >> applying clear sky mask in grid file")

            array_cs = ds_cs['clear_sky_mask'].values
            cs_lat = ds_cs['latitude'].values
            cs_lon = ds_cs['longitude'].values
            fixed_cs = fix_grid(
                array_cs,
                cs_lat,
                cs_lon,
                lats_deg,
                lons_deg,
                method='nearest')
            fixed_cs[fixed_cs > 0] = 1
            fixed_cs[np.isnan(fixed_cs)] = 1
            grid = grid * fixed_cs

        grid[grid <= 0.11] = 0  # clipping rain in .10 mm/h - 2024-02-02

        grid[np.isnan(grid)] = 0

    elif args.vname in ['10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_speed', '10m_wind_direct']:

        grid = apply_interpol(args,
                              df['latitude'].values,
                              df['longitude'].values,
                              df['altitude'].values,
                              df['data'].values,
                              lats,
                              lons,
                              topo_regrid)

        grid = fill_nans_parallel(
            grid, max_workers=4, max_distance=(500 / (112 * float(args.res))))

        if args.merge_path is not None:
            grid = merge_background(
                args, grid, lats_deg, lons_deg, topo_regrid)

        grid = savgol_filter(grid, 4, 2, axis=0)
        grid = savgol_filter(grid, 4, 2, axis=1)

        grid = level_fixer_undo(grid, topo_regrid, '10m_wind')

        grid[np.abs(grid) < 0.1] = 0
        grid[np.abs(grid) > 72] = np.nan

    elif args.vname == '10m_wind_gust':

        grid = apply_interpol(args,
                              df['latitude'].values,
                              df['longitude'].values,
                              df['altitude'].values,
                              df['data'].values,
                              lats,
                              lons,
                              topo_regrid)

        grid = fill_nans_parallel(grid, max_workers=4,
                                  max_distance=(args.interpolation_radius * 2 / (112 * float(args.res))))

        if args.spread_correction:
            cgrid = spread_correction(
                args,
                data,
                df['latitude'].values,
                df['longitude'].values,
                grid,
                lats,
                lons,
                radius=1800)

        grid = savgol_filter(grid, 4, 2, axis=0)
        grid = savgol_filter(grid, 4, 2, axis=1)

        if args.spread_correction:
            if cgrid.T == grid.shape:
                grid = grid + cgrid.T
            else:
                grid = grid + cgrid

        if args.merge_path is not None:
            grid = merge_background(
                args, grid, lats_deg, lons_deg, topo_regrid)

        grid = level_fixer_undo(grid, topo_regrid, '10m_wind')
        grid[grid < 0.1] = np.nan
        grid[grid > 72] = np.nan

    else:
        grid = apply_interpol(args,
                              df['latitude'].values,
                              df['longitude'].values,
                              df['altitude'].values,
                              df['data'].values,
                              lats,
                              lons,
                              topo_regrid)

        if args.merge_path is not None:
            grid = merge_background(
                args, grid, lats_deg, lons_deg, topo_regrid)

        grid = fill_nans_parallel(
            grid, max_distance=(args.interpolation_radius * 2 / (112 * float(args.res))))

        if args.spread_correction:
            cgrid = spread_correction(args,
                                      np.array(
                                          df['latitude'].values, dtype=float),
                                      np.array(
                                          df['longitude'].values, dtype=float),
                                      data,
                                      lats,
                                      lons,
                                      grid)

        grid = savgol_filter(grid, 4, 2, axis=0)
        grid = savgol_filter(grid, 4, 2, axis=1)

        if args.spread_correction:
            if cgrid.T == grid.shape:
                grid = grid + cgrid.T
            else:
                grid = grid + cgrid

        grid = level_fixer_undo(grid, topo_regrid, args.vname)

    if args.reinsert_point:
        grid = reinsert_points(grid, df, lats, lons, args)

    grid = grid.reshape((1, grid.shape[0], grid.shape[1]))

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    nc_file = nc.Dataset(args.out_path, 'w')
    var_out = args.vname

    if args.vname in ['10m_wind_direct']:
        var_out = "10m_wind_speed"
        grid[grid < 0] = 0

    ncBuilder.create_nc(
        nc_file,
        lats_deg,
        lons_deg,
        time=[df.sort_values('datetime').iloc[-1]['datetime']],
        vars={
            var_out: {'dims': ('time', 'latitude', 'longitude'), }
        }
    )

    ncBuilder.update_nc(nc_file, var_out, grid)
    nc_file.close()
    log.info(f"src.interpol_methods.interpol_data >> done interpolating {args.vname}")


def interpol_data_try_except(df, args, clear_sky_mask=None, use_topo=True):
    try:
        interpol_data(
            df,
            args,
            clear_sky_mask=clear_sky_mask,
            use_topo=use_topo)
    except Exception as e:
        log.error(f"src.interpol_methods.interpol_data_try_except >> error in interpol_data: {e}")
        traceback.print_exc()
