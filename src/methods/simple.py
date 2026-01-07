import logging
import numpy as np

from scipy.interpolate import griddata


def apply_simple_hgt(latitudes, longitudes, altitudes, data,
                     grid_lat, grid_lon, grid_alt, method):
    XI, YI, ZI = np.meshgrid(grid_lon, grid_lat, grid_alt)
    grid = griddata(
        (longitudes, latitudes, altitudes), data.flatten(), (XI, YI, ZI), method=method)
    return grid


def apply_simple(latitudes, longitudes, data,
                 grid_lat, grid_lon, method):

    XI, YI = np.meshgrid(grid_lon, grid_lat)
    grid = griddata(
        (longitudes, latitudes), data.flatten(), (XI, YI), method=method)
    return grid


def apply_linear(latitudes, longitudes, altitudes, data,
                 grid_lat, grid_lon, grid_alt, args):
    # if args.use_topo:
    #     return apply_simple_hgt(latitudes, longitudes, altitudes,
    #                             data, grid_lat, grid_lon, grid_alt, 'linear')
    # else:
    logging.info(f"src.methods.simple >> applying linear interpolation")
    grid = apply_simple(latitudes, longitudes,
                        data, grid_lat, grid_lon, 'linear')
    logging.info(f"src.methods.simple >> applying linear done")
    return grid


def apply_cubic(latitudes, longitudes, altitudes, data,
                grid_lat, grid_lon, grid_alt, args):
    # if :
    #     return apply_simple_hgt(latitudes, longitudes, altitudes,
    #                             data, grid_lat, grid_lon, grid_alt, 'cubic')
    # else:

    logging.info(f"src.methods.simple >> applying cubic interpolation")
    grid = apply_simple(latitudes, longitudes,
                        data, grid_lat, grid_lon, 'cubic')
    logging.info(f"src.methods.simple >> applying cubic done")
    return grid


def apply_nearest(latitudes, longitudes, altitudes, data,
                  grid_lat, grid_lon, grid_alt, args):
    # if args.use_topo:
    #     return apply_simple_hgt(latitudes, longitudes, altitudes,
    #                             data, grid_lat, grid_lon, grid_alt, 'nearest')
    # else:
    logging.info(f"src.methods.simple >> applying nearest interpolation")
    grid = apply_simple(latitudes, longitudes,
                        data, grid_lat, grid_lon, 'nearest')
    logging.info(f"src.methods.simple >> applying nearest done")
    return grid
