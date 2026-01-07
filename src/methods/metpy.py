import logging
import numpy as np
from metpy.interpolate import interpolate_to_grid


def apply_metpy(latitudes, longitudes, altitudes, data,
                grid_lat, grid_lon, grid_alt, args):
    '''
    methods:  “linear”, “nearest”, “cubic”, “rbf”, “natural_neighbor”, “barnes”, “cressman”
    '''

    method_ = 'cressman'
    if args.composed_method:
        if args.interpol_method1 == 'metpy':
            method_ = args.method_function1 if args.method_function1 is not None else method_

        elif args.interpol_method2 == 'metpy':
            method_ = args.method_function2 if args.method_function2 is not None else method_

    else:
        if args.method_function is not None:
            method_ = args.method_function

    logging.info(f"src.methods.metpy >> applying MetPy interpolation using {method_} method")

    boundary_coords = {
        'west': grid_lon.min(),
        'south': grid_lat.min(),
        'east': grid_lon.max(),
        'north': grid_lat.max()
    }

    grid_x, grid_y, grid = interpolate_to_grid(longitudes,
                                               latitudes,
                                               data,
                                               interp_type=method_,
                                               minimum_neighbors=3,
                                               hres=np.radians(args.res),
                                               boundary_coords=boundary_coords)

    logging.info(f"src.methods.metpy >> metpy interpolation done")

    return grid