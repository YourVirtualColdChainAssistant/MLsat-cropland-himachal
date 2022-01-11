import math
import random
import shapely
import numpy as np
import geopandas as gpd
from sklearn.neighbors import BallTree

from spacv.spacv import UserDefinedSCV, SKCV
from spacv.utils import convert_geoseries, geometry_to_2d
from spacv.grid_builder import assign_pt_to_grid, construct_grid, assign_systematic, assign_optimized_random


class ModifiedBlockCV(UserDefinedSCV):
    def _iter_test_indices(self, XYs):
        """
        Generates integer indices corresponding to test sets and
        training indices to be excluded from model training.

        Parameters
        ----------
        XYs : GeoSeries
            GeoSeries containing shapely Points that identify Easting
            and Northing coordinates of data points.

        Yields
        ------
        test_indices : array
            The testing set indices for that fold.
        train_exclude : array
            The training set indices to exclude for that fold.
        """
        grid = self.custom_polygons
        grid['grid_id'] = grid.fold_id
        grid_ids = np.unique(grid.grid_id)

        XYs = assign_pt_to_grid(XYs, grid, self.distance_metric)

        # Yield test indices and optionally training indices within buffer
        for grid_id in grid_ids:
            test_indices = XYs.loc[XYs['grid_id'] == grid_id].index.values
            # Remove empty grids
            if len(test_indices) < 1:
                continue
            grid_poly_buffer = grid.loc[[grid_id]].buffer(self.buffer_radius)
            test_indices, train_exclude = \
                super()._remove_buffered_indices(XYs, test_indices,
                                                 self.buffer_radius, grid_poly_buffer)
            yield test_indices, train_exclude

    def split(self, XYs):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        XYs : GeoSeries
            GeoSeries containing shapely Points that identify Easting
            and Northing coordinates of data points.

        Yields
        ------
        train : ndarray
            Training set indices for iteration.
        test : ndarray
            Testing set indices for iteration.
        """
        XYs = convert_geoseries(XYs).reset_index(drop=True)
        minx, miny, maxx, maxy = XYs.total_bounds

        buffer_radius = self.buffer_radius
        if buffer_radius > maxx - minx or buffer_radius > maxy - miny:
            raise ValueError(
                "buffer_radius too large and excludes all points. Given {}.".format(
                    self.buffer_radius
                )
            )
        indices = XYs.index.values

        for test_indices, train_excluded in self._iter_test_indices(XYs):
            # Exclude the training indices within buffer
            if train_excluded.shape:
                train_excluded = np.concatenate([test_indices, train_excluded])
            else:
                train_excluded = test_indices
            train_index = np.setdiff1d(
                np.union1d(
                    indices,
                    train_excluded
                ), np.intersect1d(indices, train_excluded)
            )
            if len(train_index) < 1:
                raise ValueError(
                    "Training set is empty. Try lowering buffer_radius to include more training instances."
                )
            test_index = indices[test_indices]
            yield train_index, test_index


class ModifiedSKCV(SKCV):
    def split(self, XYs):
        XYs = convert_geoseries(XYs).reset_index(drop=True)
        minx, miny, maxx, maxy = XYs.total_bounds

        buffer_radius = self.buffer_radius
        if buffer_radius > maxx - minx or buffer_radius > maxy - miny:
            raise ValueError(
                "buffer_radius too large and excludes all points. Given {}.".format(
                    self.buffer_radius
                )
            )
        indices = XYs.index.values

        for test_indices, train_excluded in self._iter_test_indices(XYs):
            # Exclude the training indices within buffer
            if train_excluded.shape:
                train_excluded = np.concatenate([test_indices, train_excluded])
            else:
                train_excluded = test_indices
            train_index = np.setdiff1d(
                np.union1d(
                    indices,
                    train_excluded
                ), np.intersect1d(indices, train_excluded)
            )
            if len(train_index) < 1:
                raise ValueError(
                    "Training set is empty. Try lowering buffer_radius to include more training instances."
                )
            test_index = indices[test_indices]
            yield train_index, test_index


def construct_grid_to_fold(polygons_geo, tiles_x, tiles_y, shape='square', method='random', direction='diagonal',
                           data=None, n_fold=5, n_sims=10, distance_metric='euclidean', random_state=42):
    polygons, grid = construct_valid_grid(polygons_geo, tiles_x, tiles_y, shape)
    grid = assign_grid_to_fold(polygons, grid, tiles_x, tiles_y, method=method, shape=shape, direction=direction,
                               data=data, n_fold=n_fold, n_sims=n_sims, distance_metric=distance_metric,
                               random_state=random_state)  # columns=[geometry, grid_id, fold_id]
    map_grid_to_fold = {grid_id: fold_id for grid_id, fold_id in zip(grid.grid_id, grid.fold_id)}
    polygons['fold_id'] = polygons.grid_id.map(map_grid_to_fold)  # columns=[geometry, grid_id, fold_id]
    return grid


def construct_valid_grid(polygons_geo, tiles_x, tiles_y, shape):
    polygons_gpd = gpd.GeoDataFrame([shapely.geometry.shape(poly) for poly in polygons_geo], columns=['geometry'])
    grid_all = construct_grid(polygons_gpd, tiles_x=tiles_x, tiles_y=tiles_y, shape=shape)
    grid_all['grid_id'] = grid_all.index
    polygons_gpd, grid_all = assign_polygons_to_grid(polygons_gpd, grid_all)  # columns=[geometry, grid_id]
    grid_valid = grid_all.loc[sorted(polygons_gpd.grid_id.unique().astype(int)), :].reset_index(drop=True)
    return polygons_gpd, grid_valid


def assign_polygons_to_grid(polygons, grid, distance_metric='euclidean', random_state=None):
    """
    Spatial join polygons to grids. Reassign border points to nearest grid based on centroid distance.
    """
    np.random.seed(random_state)
    # Equate spatial reference systems if defined
    if not grid.crs == polygons.crs:
        grid.crs = polygons.crs
    polygons = gpd.sjoin(polygons, grid, how='left', op='within')[['geometry', 'grid_id']]
    # In rare cases, points will sit at the border separating two grids
    if polygons['grid_id'].isna().any():
        # Find border pts and assign to nearest grid centroid
        grid_centroid = grid.geometry.centroid
        grid_centroid = geometry_to_2d(grid_centroid)
        border_polygon_index = polygons['grid_id'].isna()
        border_centroid = polygons[border_polygon_index].geometry.centroid
        border_centroid = geometry_to_2d(border_centroid)

        # Update border pt grid IDs
        tree = BallTree(grid_centroid, metric=distance_metric)
        grid_id = tree.query(border_centroid, k=1, return_distance=False).flatten()
        grid_id = grid.loc[grid_id, 'grid_id'].values
        polygons.loc[border_polygon_index, 'grid_id'] = grid_id

        # update grid shape, not split polygons
        for poly, grid_id in zip(polygons[border_polygon_index].geometry, polygons[border_polygon_index].grid_id):
            grid.loc[grid.grid_id.values == grid_id, 'geometry'] = \
                grid.loc[grid.grid_id.values == grid_id, 'geometry'].union(poly)
            grid.loc[grid.grid_id.values != grid_id, 'geometry'] = \
                grid.loc[grid.grid_id.values != grid_id, 'geometry'].difference(poly)

    return polygons, grid


def assign_grid_to_fold(polygons, grid, tiles_x, tiles_y, method='random', shape='square',
                        direction='diagonal', data=None, n_fold=5, n_sims=10,
                        distance_metric='euclidean', random_state=None):
    # Set grid assignment method
    if method == 'unique':
        grid['fold_id'] = grid.index
    elif method == 'systematic':
        if shape != 'square':
            raise Exception("systematic grid assignment method does not work for irregular grids.")
        grid['fold_id'] = assign_systematic(grid, tiles_x, tiles_y, direction)
    elif method == 'random':
        grid['fold_id'] = assign_random(grid, n_fold, random_state)
    elif method == 'optimized_random':
        grid['fold_id'] = assign_optimized_random(grid, polygons, data, n_fold, n_sims, distance_metric)
    else:
        raise ValueError("Method not recognised. Choose between: unique, systematic, random or optimized_random.")
    return grid


def assign_random(grid, n_fold, random_state):
    random.seed(random_state)

    n_grids = grid.shape[0]
    n_reps = math.floor(n_grids / n_fold)

    idx = np.arange(n_grids)
    random.shuffle(idx)

    val = np.repeat(np.arange(n_fold), n_reps)
    for _ in range(n_grids - val.shape[0]):
        val = np.insert(val, -1, n_fold - 1, axis=0)

    grid_id = np.empty(n_grids).astype(int)
    grid_id[idx] = val

    return grid_id
