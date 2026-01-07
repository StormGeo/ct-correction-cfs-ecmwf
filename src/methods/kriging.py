import logging
import numpy as np

from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

def _uni_kriging(latitudes, longitudes, altitudes, data, glats, glons,
                 max_dist=2, topo=None, method='linear'):

    z = data

    gridx, gridy = np.meshgrid(glons, glats)

    # creating mask array
    mask = np.zeros(gridx.shape, dtype=bool)

    count = 0
    for xi, yi in zip(longitudes, latitudes):
        distance = np.sqrt((gridx - xi) ** 2 + (gridy - yi) ** 2)
        mask |= distance < max_dist


        count += 1

    UK = UniversalKriging(longitudes, latitudes, z,
                          variogram_model=method,
                          drift_terms='regional_linear',
                          specified_drift=altitudes,
                          # external_drift=topo,
                          # drift_terms=['external_Z'],
                          # external_drift_x=glons,
                          # external_drift_y=glats,
                          exact_values=False,
                          enable_plotting=False,
                          verbose=False,
                          pseudo_inv=True)

    z_interp, ss = UK.execute('masked', glons, glats, mask=~mask, specified_drift_arrays=topo)

    z_interp[~mask] = np.nan

    return z_interp


def _kriging(latitudes, longitudes, altitudes, data, glats, glons,
             max_dist=2, topo=None, max_topo_dif=50, method='linear'):

    z = data

    gridx, gridy = np.meshgrid(glons, glats)

    # creating mask array
    mask = np.zeros(gridx.shape, dtype=bool)

    count = 0
    for xi, yi in zip(longitudes, latitudes):
        distance = np.sqrt((gridx - xi) ** 2 + (gridy - yi) ** 2)
        if topo is None:
            mask |= distance < max_dist

        else:
            diff_topo = np.abs(topo - altitudes[count])
            if altitudes[count] < 500:
                mask |= (distance < max_dist) & (diff_topo <= max_topo_dif)
            else:
                mask |= distance < max_dist

        count += 1

    # # Check dimensions of the grid and the mask
    # print(f"Grid dimensions: {gridx.shape}, {gridy.shape}")
    # print(f"Mask dimensions: {mask.shape}")

    # perform Kriging interpolation
    OK = OrdinaryKriging(np.rad2deg(longitudes),
                         np.rad2deg(latitudes),
                         z,
                         variogram_model=method,
                         verbose=False,
                         enable_plotting=False,
                         exact_values=False,
                         coordinates_type='geographic',
                         pseudo_inv=False)

    # using mask to restrict the region
    z_interp, ss = OK.execute('masked', np.rad2deg(glons), np.rad2deg(glats), mask=~mask)
    z_interp[~mask] = np.nan
    return z_interp


def apply_kriging(latitudes, longitudes, altitudes, data,
                  grid_lat, grid_lon, grid_alt, args):

    max_dist = np.radians(args.interpolation_radius / 111.32)
    method_ = 'linear'

    if args.composed_method:
        if args.interpol_method1 == 'kriging':
            method_ = args.method_function1 if args.method_function1 is not None else method_

        elif args.interpol_method2 == 'kriging':
            method_ = args.method_function2 if args.method_function2 is not None else method_

    else:
        if args.method_function is not None:
            method_ = args.method_function

    logging.info(f"src.methods.kriging >> applying kriging interpolation @ using {method_} method")

    grid = _kriging(latitudes, longitudes, altitudes, data, grid_lat, grid_lon,
                    max_dist=max_dist, topo=grid_alt, method=method_)

    logging.info(f"src.methods.kriging >> kriging interpolation done")

    return grid

def apply_universal_kriging(latitudes, longitudes, altitudes, data,
                            grid_lat, grid_lon, grid_alt, args):
    max_dist = np.radians(args.interpolation_radius / 111.32)
    method_ = 'linear'
    if args.composed_method:
        if args.interpol_method1 == 'ukriging':
            method_ = args.method_function1 if args.method_function1 is not None else method_

        elif args.interpol_method2 == 'ukriging':
            method_ = args.method_function2 if args.method_function2 is not None else method_

    else:
        if args.method_function is not None:
            method_ = args.method_function

    logging.info(f"src.methods.kriging >> applying universal kriging interpolation @ using {method_} method")

    grid = _uni_kriging(latitudes, longitudes, altitudes, data, grid_lat, grid_lon,
                    max_dist=max_dist, topo=grid_alt, method=method_)

    logging.info(f"src.methods.kriging >> universal kriging interpolation done")

    return grid
