import numpy as np
import logging
from scipy.interpolate import RegularGridInterpolator


def _bilinear(latitudes, longitudes, data,
              grid_lat, grid_lon):
    f = RegularGridInterpolator((longitudes, latitudes), data)
    X, Y = np.meshgrid(grid_lon, grid_lat)
    grid = f((X, Y))
    return grid


def _bilinear_hgt(latitudes, longitudes, altitudes, data,
                  grid_lat, grid_lon, grid_alt):
    f = RegularGridInterpolator((longitudes, latitudes, altitudes), data)
    X, Y, Z = np.meshgrid(grid_lon, grid_lat, grid_alt)
    grid = f((Y, X, Z))
    return grid


def apply_bilinear(latitudes, longitudes, altitudes, data,
                   grid_lat, grid_lon, grid_alt, args):
    # if args.use_topo:
    #     return _bilinear_hgt(latitudes, longitudes, altitudes,
    #                          data, grid_lat, grid_lon, grid_alt)
    # else:

    logging.info(f"src.methods.bilinear >> applying bilinear interpolation")

    grid = _bilinear(latitudes, longitudes, data,
                      grid_lat, grid_lon)

    logging.info(f"src.methods.bilinear >> bilinear interpolation done")
    return grid
