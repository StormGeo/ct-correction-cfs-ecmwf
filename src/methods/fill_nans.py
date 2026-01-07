import os

import logging as log

import concurrent.futures

import numpy as np

from scipy.signal import savgol_filter
from scipy.spatial import cKDTree


def get_max_pool_size(load_avg=16):
    pool_size = int(os.cpu_count() -
                    ((os.cpu_count() * os.getloadavg()[0]) / load_avg))
    if pool_size < 2:
        return 2
    else:
        return pool_size


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    sin_dlat_2 = np.sin(dlat / 2)
    sin_dlon_2 = np.sin(dlon / 2)

    a = sin_dlat_2**2 + np.cos(lat1) * np.cos(lat2) * sin_dlon_2**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def idw(x, y, values, target_x, target_y, power=1,
        method='euclidian', max_distance=None):
    if method == 'euclidian':
        distances = np.sqrt((x - target_x)**2 + (y - target_y)**2)
    else:
        distances = haversine(x, y, target_x, target_y)
    if max_distance:
        distances[distances > max_distance] = np.nan
    weights = 1.0 / (distances ** power)
    weights /= np.sum(weights)
    interpolated_value = np.dot(values, weights)
    return interpolated_value


def fill_nans_parallel(grid, max_workers=12, max_distance=50, weighted=False):
    nan_mask = np.isnan(grid) | np.isinf(grid)
    total_nan = np.count_nonzero(nan_mask)
    total_val = np.count_nonzero(~nan_mask)

    if total_nan == 0:
        return grid
    perc = total_nan / (total_val + total_nan)
    if perc < 0.01:
        return grid

    log.info(f'src.fill_nans >> still missing: {total_nan}')
    log.info('src.fill_nans >> filling missing values with neighbors')

    non_nan_indices = np.where(~nan_mask)
    x = non_nan_indices[0]
    y = non_nan_indices[1]

    nan_indices = np.where(nan_mask)
    nan_x = nan_indices[0]
    nan_y = nan_indices[1]

    values = grid[~nan_mask]

    try:
        kdtree = cKDTree(list(zip(x, y)))
    except:
        return grid

    def _fill_nan(i, j):
        dist, nearest_idx = kdtree.query((i, j), k=1)

        if not weighted:

            nearest_x, nearest_y = x[nearest_idx], y[nearest_idx]
            v = grid[nearest_x, nearest_y]

        else:
            if dist < 8:
                nearest_x, nearest_y = x[nearest_idx], y[nearest_idx]
                v = grid[nearest_x, nearest_y]
            elif dist <= max_distance:
                v = idw(x, y, values, i, j, power=2, method='euclidian')
                # v = idw(x, y, values, i, j, power=2, method='haversine')
            else:
                v = np.nan

        return (i, j, v)

    available_workers = get_max_pool_size(load_avg=max_workers * 1 / 2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=available_workers) as executor:
        futures = []
        for ij in zip(nan_x, nan_y):
            future = executor.submit(_fill_nan, *ij)
            futures.append(future)
        concurrent.futures.wait(futures)

    for future in futures:
        i, j, v = future.result()
        grid[i, j] = v

    log.info('src.fill_nans >> done filling missing values')
    return grid
