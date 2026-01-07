import logging

import numpy as np

R = 6371  # Earth radius in km


def great_circle_distance(latitudes, longitudes,
                          altitudes, grid_lat, grid_lon, grid_alt):

    lat1, lon1, alt1 = latitudes, longitudes, altitudes
    lat2, lon2, alt2 = grid_lat, grid_lon, grid_alt

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    dalt = (alt2 - alt1) / 1000.0  # Convert to km

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    d = np.sqrt(d**2 + dalt**2)
    return d


def cressman_weights(latitudes, longitudes, altitudes,
                     grid_lat, grid_lon, grid_alt, radius, use_topo):

    if not use_topo:
        dist = np.sqrt((latitudes - grid_lat)**2 + (longitudes - grid_lon)**2)
        radius = np.radians(radius / 111.32)
    else:
        dist = great_circle_distance(latitudes,
                                     longitudes,
                                     altitudes,
                                     grid_lat,
                                     grid_lon,
                                     grid_alt)

    weights = (radius**2 - dist**2) / (radius**2 + dist**2)
    weights[dist > radius] = 0
    return weights


def apply_cressman_loop(latitudes, longitudes, altitudes, data,
                        grid_lat, grid_lon, grid_alt,
                        radius=500, use_topo=False):
    grid = np.full((len(grid_lat), len(grid_lon)), np.nan)
    for i in range(len(grid_lat)):
        for j in range(len(grid_lon)):
            weights = cressman_weights(latitudes, longitudes, altitudes,
                                       grid_lat[i], grid_lon[j], grid_alt[i, j], radius)
            grid[i, j] = np.sum(data * weights) / np.sum(weights)
    return grid


def apply_cressman(latitudes, longitudes, altitudes, data,
                   grid_lat, grid_lon, grid_alt, args):

    radius = args.interpolation_radius

    logging.info(f"src.methods.cressman >> applying cressman interpolation using {radius}km of radius")

    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    latitudes = np.asarray(latitudes)[:, np.newaxis, np.newaxis]
    longitudes = np.asarray(longitudes)[:, np.newaxis, np.newaxis]
    altitudes = np.asarray(altitudes)[:, np.newaxis, np.newaxis]

    weights = cressman_weights(latitudes, longitudes, altitudes,
                               grid_lat, grid_lon, grid_alt, radius, False)

    weighted_data_sum = np.sum(
        weights *
        data[
            :,
            np.newaxis,
            np.newaxis],
        axis=0)

    total_weight = np.sum(weights, axis=0)

    grid = np.divide(weighted_data_sum, total_weight, where=total_weight > 1)
    grid[total_weight == 0] = np.nan

    logging.info(f"src.methods.cressman >> cressman interpolation done")

    return grid
