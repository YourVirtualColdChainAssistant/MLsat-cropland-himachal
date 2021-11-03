import numpy as np
import warnings
import geopandas as gpd
from spacv.grid_builder import assign_pt_to_grid, construct_grid, assign_randomized, assign_systematic, \
    assign_optimized_random
from spacv.base_classes import BaseSpatialCV
from spacv.utils import convert_geoseries
from sklearn.model_selection import GroupKFold


# TODO: do not split polygons in block CV
class ModifiedBlockCV(BaseSpatialCV):
    """
    Spatial cross-validation using user-defined polygons.

    Yields indices to split data into training and test sets.

    Parameters
    ----------
    tiles_x : integer, default=5
        Number of grid tiles in the West-East direction.
    tiles_y : integer, default=5
        Number of grid tiles in the North-South direction.
    shape : string, default='square'
        Specify shape of grid polygons, square or hex.
    method : string, default='unique'
        Choose grid ID assignment method to build folds. Options
        are: unique, where every polygon in the grid is a fold;
        systematic, where folds reflect diagonal or anti-diagonal
        patterns across the study area; random, where folds are
        randomly assigned into groups determined by n_groups parameter;
        and optimized_random, where random assignment of grids into
        groups are optimized by reducing disimilarity between folds.
    buffer_radius : integer, default=0
        Buffer radius (dead zone) to exclude training points that are
        within a defined distance of test data within a fold.
    direction : string, default='diagonal'
        Choose direction of pattern for systematic grid assignment,
        diagonal or anti-diagonal (anti).
    n_groups : integer, default=5
        Number of folds to randomly assign grid polygons into.
    data : array
        Array containing covariates used in predictive task. Used to
        calculate disimilarity of feature space between folds to
        find the optimized random grid assignment.
    n_sims : integer, default=10
        Number of iterations in which to find optimized grid assignment
        into folds.
    distance_metric : string, default='euclidean'
        Distance metric used to reconcile points that sit at exact
        border between two grids. Defaults to euclidean assuming
        projected coordinate system, otherwise use haversine for
        unprojected spaces.
    random_state : int, default=None
        random_state is the seed used by the random number generator.

    Yields
    ------
    test_indices : array
        The testing set indices for that fold.
    train_exclude : array
        The training set indices to exclude for that fold.

    """

    def __init__(
            self,
            tiles_x=5,
            tiles_y=5,
            shape='square',
            method='unique',
            buffer_radius=0,
            direction='diagonal',
            n_groups=5,
            data=None,
            n_sims=10,
            distance_metric='euclidean',
            random_state=None
    ):
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.shape = shape
        self.method = method
        self.buffer_radius = buffer_radius
        self.direction = direction
        self.n_groups = n_groups
        self.data = data
        self.n_sims = n_sims
        self.distance_metric = distance_metric
        self.n_splits = tiles_x * tiles_y
        self.random_state = random_state

    def construct_valid_block(self, XYs):
        # Define grid type used in CV procedure but with only valid blocks
        grid_all = construct_grid(XYs, tiles_x=self.tiles_x, tiles_y=self.tiles_y, shape=self.shape)
        grid_all['grid_id'] = grid_all.index

        # Convert to GDF to use Geopandas functions
        XYs = gpd.GeoDataFrame(({'geometry': XYs}))

        # Find the valid grid with points falling in
        XYs_assigned = assign_pt_to_grid(XYs, grid_all, self.distance_metric)
        grid_valid = grid_all.loc[sorted(XYs_assigned.grid_id.unique().astype(int)), :].reset_index(drop=True)
        grid_valid = grid_valid.rename(columns={'grid_id': 'grid_id_valid'})

        # assign
        grid = assign_grid_to_fold(XYs, grid_valid, self.tiles_x, self.tiles_y, method=self.method, shape=self.shape,
                                   direction=self.direction, data=self.data, n_groups=self.n_groups, n_sims=self.n_sims,
                                   distance_metric=self.distance_metric, random_state=self.random_state)
        XYs = assign_pt_to_grid(XYs, grid, self.distance_metric)
        grid_ids = np.unique(grid.grid_id)

        return XYs, grid, grid_ids

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
        XYs, grid, grid_ids = self.construct_valid_block(XYs)

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


def assign_grid_to_fold(XYs, grid, tiles_x, tiles_y, method='unique', shape='square',
                        direction='diagonal', data=None, n_groups=5, n_sims=10,
                        distance_metric='euclidean', random_state=None):
    # Set grid assignment method
    if method == 'unique':
        grid['grid_id'] = grid.index
    elif method == 'systematic':
        if shape != 'square':
            raise Exception("systematic grid assignment method does not work for irregular grids.")
        grid['grid_id'] = assign_systematic(grid, tiles_x, tiles_y, direction)
    elif method == 'random':
        grid['grid_id'] = assign_randomized(grid, n_groups, random_state)
    elif method == 'optimized_random':
        grid['grid_id'] = assign_optimized_random(grid, XYs, data,
                                                  n_groups,
                                                  n_sims,
                                                  distance_metric)
    else:
        raise ValueError("Method not recognised. Choose between: unique, systematic, random or optimized_random.")
    return grid
