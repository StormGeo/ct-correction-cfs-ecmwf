import logging as log

import numpy as np

EPSILON = 1e-3


class Spatial_Circle:

    def __init__(self, target_shape, radius, res):
        self.DKM = 0.009
        self.target_shape = target_shape
        self.radius = radius
        self.res = res
        self.weight = None
        self.max_idx = None
        self.gen_area()
        # self.gen_weights()

    def gen_area(self):
        dg = self.DKM * self.radius
        self.max_idx = int(round(dg / self.res))
        # self.max_idx = int(dg / self.res) + 1

        self.weight = np.full(
            (2 * self.max_idx + 1, 2 * self.max_idx + 1), np.nan)

    def gen_weights(self):
        radius2 = self.radius**2
        dist = lambda i, j: np.sqrt(
            (i - self.max_idx)**2 + (j - self.max_idx)**2) * self.res / self.DKM
        mask = np.where(np.full(self.weight.shape, True))

        for (i, j) in zip(*mask):
            x = dist(i, j)
            x2 = x**2
            if (radius2 - x2) > EPSILON:
                weight = (radius2 - x2) / (radius2 + x2)
            else:
                weight = 0
            if weight >= 0:
                self.weight[i, j] = weight

    def gen_weights_sqr_dist(self):
        dist = lambda i, j: np.sqrt(
            (i - self.max_idx)**2 + (j - self.max_idx)**2) * self.res / self.DKM
        mask = np.where(np.full(self.weight.shape, True))

        for (i, j) in zip(*mask):
            x = dist(i, j)
            if x < 1:
                self.weight[i, j] = 1
            elif x <= self.radius:
                self.weight[i, j] = 1 / x**2
            else:
                self.weight[i, j] = 0

    def gen_weights_sqr_dist_hgt(self, topo, altitude, alpha=150):
        radius2 = self.radius**2
        dist = lambda i, j: np.sqrt(
            (i - self.max_idx)**2 + (j - self.max_idx)**2) * self.res / self.DKM
        mask = np.where(np.full(self.weight.shape, True))

        for (i, j) in zip(*mask):
            x = dist(i, j)
            ii = i if i < topo.shape[0] else topo.shape[0] - 1
            jj = j if j < topo.shape[1] else topo.shape[1] - 1

            alt_diff = (topo[ii, jj] - altitude) / alpha
            d_3d = np.sqrt(x**2 + alt_diff**2)  # 3D distance

            if d_3d < 1:
                self.weight[i, j] = 1
            elif x <= self.radius:
                self.weight[i, j] = 1 / d_3d**2
            else:
                self.weight[i, j] = 0

    def gen_weights_hgt(self, topo, altitude, alpha=150):
        radius2 = self.radius**2
        dist = lambda i, j: np.sqrt(
            (i - self.max_idx)**2 + (j - self.max_idx)**2) * self.res / self.DKM
        mask = np.where(np.full(self.weight.shape, True))

        for (i, j) in zip(*mask):
            x = dist(i, j)
            ii = i if i < topo.shape[0] else topo.shape[0] - 1
            jj = j if j < topo.shape[1] else topo.shape[1] - 1

            alt_diff = (topo[ii, jj] - altitude) / alpha
            d_3d = np.sqrt(x**2 + alt_diff**2)  # 3D distance

            if (self.radius - d_3d) > EPSILON:
                if d_3d < 1:
                    self.weight[i, j] = 1
                if d_3d <= self.radius:
                    w = (radius2 - d_3d**2) / (radius2 + d_3d**2)
                    if w >= 0:
                        self.weight[i, j] = w
                    else:
                        self.weight[i, j] = 0
                else:
                    self.weight[i, j] = 0
            else:
                self.weight[i, j] = (
                    radius2 - EPSILON**2) / (radius2 + EPSILON**2)

    def interpolate_circle(self, ilats, ilons, values):
        exp_target = np.full((self.target_shape[0] + 2 * self.max_idx,
                              self.target_shape[1] + 2 * self.max_idx), np.nan)
        exp_weight = exp_target.copy()

        for i, j, v in zip(ilats, ilons, values):
            l, r = slice(i, i + 2 * self.max_idx +
                         1), slice(j, j + 2 * self.max_idx + 1)
            exp_target[l, r] = np.nansum(
                [exp_target[l, r], self.weight * v], axis=0)
            exp_weight[l, r] = np.nansum(
                [exp_weight[l, r], self.weight], axis=0)

        return (exp_target / exp_weight)[self.max_idx:-
                                         self.max_idx, self.max_idx:-self.max_idx]

    def interpolate_circle_hgt(
            self, ilats, ilons, values, altitudes, topo, mode):
        exp_target = np.full((self.target_shape[0] + 2 * self.max_idx,
                              self.target_shape[1] + 2 * self.max_idx), np.nan)
        exp_weight = exp_target.copy()

        for i, j, v, alt in zip(ilats, ilons, values, altitudes):
            l, r = slice(i, i + 2 * self.max_idx +
                         1), slice(j, j + 2 * self.max_idx + 1)
            # Pass the altitude of the current observation
            if mode == 'cres':
                self.gen_weights_hgt(topo[l, r], alt)
            else:
                self.gen_weights_sqr_dist_hgt(topo[l, r], alt)
            exp_target[l, r] = np.nansum(
                [exp_target[l, r], self.weight * v], axis=0)
            exp_weight[l, r] = np.nansum(
                [exp_weight[l, r], self.weight], axis=0)

        return (exp_target / exp_weight)[self.max_idx:-
                                         self.max_idx, self.max_idx:-self.max_idx]


def interpolate_circles(x, y, topo, values, target_x, target_y, grid_alt, args,
                        max_distance=800, res=0.125):

    ilat = [np.argmin(abs(i - target_x)) for i in x]
    ilon = [np.argmin(abs(i - target_y)) for i in y]
    tx_size = target_x.size
    ty_size = target_y.size
    res = float(res)

    if args.vname == 'total_precipitation':
        previous_shape = (tx_size, ty_size)
        distances = np.arange(5, max_distance, 10)
        grids = []
        for distance in distances:
            circle = Spatial_Circle(
                (tx_size, ty_size), distance, res)
            circle.gen_weights()
            grid = circle.interpolate_circle(ilat, ilon, values)
            if previous_shape == grid.shape:
                grids.append(grid)

        grids = np.array(grids)
        grid = np.nanmean(grids, axis=0)
    else:
        distances = np.round(
            np.geomspace(
                res * 112,
                max_distance,
                10,
                endpoint=True),
            decimals=0)
        previous_shape = (tx_size, ty_size)
        grids = []
        for distance in distances:
            circle = Spatial_Circle(
                (tx_size, ty_size), distance, res)
            circle.gen_weights()
            grid = circle.interpolate_circle(ilat, ilon, values)

            if previous_shape == grid.shape:
                grids.append(grid)
        grids = np.array(grids)
        grid = np.nanmedian(grids, axis=0)

    return grid


def interpolate_circles_hgt(x, y, values, target_x, target_y, vname,
                            topo, altitudes, max_distance=800, res=0.125, mode='cres'):
    log.info('Starting interpolation')
    ilat = [np.argmin(abs(i - target_x)) for i in x]
    ilon = [np.argmin(abs(i - target_y)) for i in y]
    tx_size = target_x.size
    ty_size = target_y.size
    res = float(res)

    distances = np.round(
        np.geomspace(
            res * 112,
            max_distance,
            10,
            endpoint=True),
        decimals=0)
    previous_shape = (tx_size, ty_size)
    grids = []
    for distance in distances:
        circle = Spatial_Circle(
            (tx_size, ty_size), distance, res)
        grid = circle.interpolate_circle_hgt(ilat,
                                             ilon,
                                             values,
                                             altitudes,
                                             topo,
                                             mode)

        if previous_shape == grid.shape:
            grids.append(grid)
    grids = np.array(grids)
    grid = np.nanmedian(grids, axis=0)
    # grid = np.full(grids[0].shape, np.nan)
    # for g in grids:
    #     grid[np.isnan(grid)] = g[np.isnan(grid)]
    # grid = np.nanmean([bgrid, grid], axis=0)
    return grid


def map_heights_to_range(heights, input_min=0,
                         input_max=5000, output_min=0, output_max=35):
    heights = np.clip(heights, input_min, input_max)

    # Normalize and map the input values to the new range
    scaled_values = (heights - input_min) / (input_max - input_min)
    mapped_values = output_min + scaled_values * (output_max - output_min)

    return np.round(mapped_values).astype(int)


def apply_circles(latitudes, longitudes, altitudes, data,
                  grid_lat, grid_lon, grid_alt, args):
    # if args.use_topo:
    #     return interpolate_circles_hgt(latitudes, longitudes, data, args.vnane,
    #                                    grid_alt, altitudes, args.interpolation_radius, args.resolution)
    # else:
    log.info(f"src.methods.spatial_circles >> applying circles interpolation")

    grid = interpolate_circles(latitudes, longitudes, altitudes, data, grid_lat, grid_lon, grid_alt, args,
                               max_distance=args.interpolation_radius, res=args.res)
    # grid = interpolate_circles(latitudes, longitudes, data, args.vname,
    #                             grid_lat, grid_lon, args.interpolation_radius, args.res)

    log.info(f"src.methods.spatial_circles >> circles interpolation done")
    return grid
