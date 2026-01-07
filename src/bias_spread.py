import logging as log

import numpy as np

from scipy.signal import savgol_filter


def decay_circle(size, center_val, border_val):
    size = int(size)
    matrix = np.full((size, size), np.nan)
    radius = size // 2
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - radius)**2 + (j - radius)**2)
            if dist > radius:
                matrix[i, j] = np.nan
            else:
                matrix[i, j] = center_val - \
                    ((center_val - border_val) *
                     (float(dist) / float(radius)))
    return matrix


def spread_correction(args, data, observed_latitudes,
                      observed_longitudes, grid, lats, lons, radius=1800):

    log.info('src.bias_spread >> applying local bias correction')
    stencil_size = int((radius / (112 * float(args.res))) / 2 + 1)

    if stencil_size < 7:
        stencil_size = 7
    nlats, nlons = len(lats), len(lons)
    cgrid = np.full((nlats, nlons), np.nan)
    ilat = [np.argmin(abs(i - lats)) for i in observed_latitudes]
    ilon = [np.argmin(abs(i - lons)) for i in observed_longitudes]
    for da, ix, iy in zip(data, ilat, ilon):
        if (iy > lats.size) or (ix > lons.size):
            continue
        dif = da - grid[ix, iy]
        if np.abs(dif) > 1e-1:
            circle = decay_circle((2 * stencil_size) + 1, dif, 0)

            if ix < stencil_size:
                if iy < stencil_size:
                    circle = circle[stencil_size - ix:, stencil_size - iy:]
                else:
                    circle = circle[stencil_size - ix:, :]
            if iy < stencil_size:
                circle = circle[:, stencil_size - iy:]

            if (nlats - ix) < stencil_size:
                if (nlons - iy) < stencil_size:
                    circle = circle[:(nlats - ix), :(nlons - iy)]
                else:
                    circle = circle[:(nlats - ix), :]
            if (nlons - iy) < stencil_size:
                circle = circle[:, :(nlons - iy)]

            stencil = cgrid[
                ix - stencil_size - 1:ix + stencil_size,
                iy - stencil_size - 1:iy + stencil_size].copy()

            if np.all(np.isnan(stencil)):
                try:
                    cgrid[ix -
                          stencil_size -
                          1:ix +
                          stencil_size,
                          iy -
                          stencil_size -
                          1:iy +
                          stencil_size] = circle
                except:
                    pass
            else:
                try:
                    patch = circle.copy()
                    patch[~np.isnan(stencil)] = np.nanmean(
                        [stencil[~np.isnan(stencil)], circle[~np.isnan(stencil)]], axis=0)
                    cgrid[
                        ix -
                        stencil_size -
                        1:ix +
                        stencil_size,
                        iy -
                        stencil_size -
                        1:iy +
                        stencil_size] = patch
                except:
                    pass

    cgrid[np.isnan(cgrid)] = 0.
    cgrid = savgol_filter(cgrid, int(stencil_size / 2), 3, axis=0)
    cgrid = savgol_filter(cgrid, int(stencil_size / 2), 3, axis=1)
    cgrid[np.abs(cgrid) < 1e-2] = 0.
    log.info('src.bias_spread >> done with bias correction')
    return cgrid
