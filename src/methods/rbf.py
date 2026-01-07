import logging

import numpy as np

from scipy.interpolate import Rbf


def _rbf_hgt(latitudes, longitudes, altitudes, data,
             grid_lat, grid_lon, grid_alt, method='multiquadric'):
    X, Y, Z = np.meshgrid(longitudes, latitudes, altitudes)
    XI, YI, ZI = np.meshgrid(grid_lon, grid_lat, grid_alt)
    rbf = Rbf(X, Y, Z, data, method=method)
    grid = rbf(XI, YI, ZI)
    return grid


# def _rbf(latitudes, longitudes, data,
#          grid_lat, grid_lon, method='multiquadric'):
#
#     XI, YI = np.meshgrid(grid_lon, grid_lat)
#     rbf = Rbf(longitudes, latitudes, data, method=method)
#     grid = rbf(XI, YI)
#     return grid

def _rbf(latitudes, longitudes, data,
         grid_lat, grid_lon, method='multiquadric'):

    # coordinates normalization can help
    lat_norm = (latitudes - np.min(grid_lat)) / (np.max(grid_lat) - np.min(grid_lat))
    lon_norm = (longitudes - np.min(grid_lon)) / (np.max(grid_lon) - np.min(grid_lon))

    grid_lat_norm = (grid_lat - np.min(grid_lat)) / (np.max(grid_lat) - np.min(grid_lat))
    grid_lon_norm = (grid_lon - np.min(grid_lon)) / (np.max(grid_lon) - np.min(grid_lon))

    # Criar a malha de interpolação
    XI, YI = np.meshgrid(grid_lon_norm, grid_lat_norm)

    # Interpolação RBF
    rbf = Rbf(lon_norm, lat_norm, data + 273.15, function=method, smooth=0.1)
    grid = rbf(XI, YI)
    return grid - 273.15

def apply_rbf(latitudes, longitudes, altitudes, data,
              grid_lat, grid_lon, grid_alt, args, method='linear'):
    '''
    method options multiquadric, thin_plate and gaussian
    '''

    # if args.use_topo:
    #     return apply_rbf_hgt(latitudes, longitudes, altitudes,
    #                          data, grid_lat, grid_lon, grid_alt, method=method)
    # else:

    method_ = 'linear'
    if args.composed_method:
        if args.interpol_method1 == 'rbf':
            method_ = args.method_function1 if args.method_function1 is not None else method_

        elif args.interpol_method2 == 'rbf':
            method_ = args.method_function2 if args.method_function2 is not None else method_

    else:
        if args.method_function is not None:
            method_ = args.method_function

    logging.info(f"src.methods.rbf >> applying RBF interpolation @ using {method_} method")

    grid = _rbf(latitudes, longitudes, data,
                grid_lat, grid_lon, method=method_)

    logging.info(f"src.methods.rbf >> RBF interpolation done")

    return grid


def apply_tbs(latitudes, longitudes, altitudes, data,
              grid_lat, grid_lon, grid_alt, args, method='linear'):
    '''
    method options multiquadric, thin_plate and gaussian
    '''

    method_ = 'thin_plate'

    logging.info(f"src.methods.rbf >> applying TBS interpolation @ using {method_} method")

    grid = _rbf(latitudes, longitudes, data,
                grid_lat, grid_lon, method=method_)

    logging.info(f"src.methods.rbf >> TBS interpolation done")

    return grid


