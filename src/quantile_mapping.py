import logging as log

import numpy as np
import scipy.stats as stats

from scipy.interpolate import interp1d


def quantile_mapping(obs_data, grid_data):
    """
    Realiza a correção de viés baseada em Quantile Mapping.
    """
    # Ordena os dados observados e da grade
    obs_sorted = np.sort(obs_data)
    grid_sorted = np.sort(grid_data)

    # Calcula os percentis dos dados da grade
    grid_perc = np.arange(1, len(grid_sorted) + 1) / (len(grid_sorted) + 1)

    # Interpola os valores observados nos percentis da grade
    interp_func = interp1d(
        grid_perc,
        obs_sorted,
        bounds_error=False,
        fill_value="extrapolate")

    # Aplica a correção de quantil aos dados da grade
    corrected_grid_data = interp_func(
        stats.rankdata(grid_data) / (len(grid_data) + 1))

    return corrected_grid_data


def spread_correction(args, data, observed_latitudes,
                      observed_longitudes, grid, lats, lons, radius=1800):
    log.info('Applying local bias correction using Quantile Mapping')

    stencil_size = int((radius / (112 * float(args.res))) / 2 + 1)
    if stencil_size < 7:
        stencil_size = 7

    nlats, nlons = len(lats), len(lons)
    cgrid = np.full((nlats, nlons), np.nan)

    ilat = [np.argmin(abs(i - lats)) for i in observed_latitudes]
    ilon = [np.argmin(abs(i - lons)) for i in observed_longitudes]

    for da, ix, iy in zip(data, ilat, ilon):
        if (iy >= nlats) or (ix >= nlons):
            continue

        local_grid = grid[max(0, ix - stencil_size):min(nlats, ix + stencil_size + 1),
                          max(0, iy - stencil_size):min(nlons, iy + stencil_size + 1)]

        local_obs = da

        if local_grid.size > 0:
            local_grid_flat = local_grid.flatten()
            local_obs_flat = np.full(local_grid_flat.shape, local_obs)

            corrected_values = quantile_mapping(
                local_obs_flat, local_grid_flat)

            corrected_grid = corrected_values.reshape(local_grid.shape)

            cgrid[max(0, ix - stencil_size):min(nlats, ix + stencil_size + 1),
                  max(0, iy - stencil_size):min(nlons, iy + stencil_size + 1)] = corrected_grid

    # Preserva valores da grade original onde não houve observações
    cgrid[np.isnan(cgrid)] = grid[np.isnan(cgrid)]
    log.info('Done with bias correction')
    return cgrid
