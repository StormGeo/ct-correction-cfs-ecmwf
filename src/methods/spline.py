import numpy as np

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import SmoothBivariateSpline


def _spline_hgt(latitudes, longitudes, altitudes, data,
                grid_lat, grid_lon, grid_alt, degree=3):
    # TODO: Implement 3D spline interpolation, this snippet only implements 2D spline interpolation
    # Assuming HGT is a dimension of the data array (pressure level)
    X, Y, Z = np.meshgrid(longitudes, latitudes, altitudes)
    XI, YI, ZI = np.meshgrid(grid_lon, grid_lat, grid_alt)

    grid = np.zeros_like(XI)

    for i, alt in enumerate(altitudes):
        spline = RectBivariateSpline(
            latitudes, longitudes, data[
                :, :, i], kx=degree, ky=degree)
        grid[:, :, i] = spline(grid_lat, grid_lon)

    return grid


def _spline(latitudes, longitudes, data,
            grid_lat, grid_lon, degree=3):
    XI, YI = np.meshgrid(grid_lon, grid_lat)

    spline = SmoothBivariateSpline(
        latitudes,
        longitudes,
        data.flatten(),
        kx=degree,
        ky=degree)
    grid = spline.ev(XI, YI)

    return grid


def apply_spline(latitudes, longitudes, altitudes, data,
                 grid_lat, grid_lon, grid_alt, args):

    return _spline(latitudes, longitudes,
                   data, grid_lat, grid_lon)
