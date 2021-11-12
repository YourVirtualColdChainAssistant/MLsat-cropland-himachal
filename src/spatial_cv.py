import numpy as np
import warnings
import geopandas as gpd
from spacv.grid_builder import assign_pt_to_grid
from spacv.base_classes import BaseSpatialCV
from spacv.utils import convert_geoseries
from spacv.spacv import UserDefinedSCV, SKCV


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
        num_samples = XYs.shape[0]
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
        num_samples = XYs.shape[0]
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
