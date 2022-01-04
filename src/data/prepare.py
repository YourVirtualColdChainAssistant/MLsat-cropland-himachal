import os
import math
import random
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from src.data.engineer import add_bands, get_raw_every_n_weeks, get_statistics, get_difference, \
    get_all_spatial_features
from src.evaluation.visualize import plot_timestamps, plot_profile
from src.utils.util import count_classes, load_shp_to_array, multipolygons_to_polygons, \
    prepare_meta_window_descriptions, prepare_meta_descriptions
from src.utils.stack import stack_timestamps
from scipy import interpolate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from spacv.utils import geometry_to_2d
from sklearn.neighbors import BallTree
from spacv.grid_builder import construct_grid, assign_systematic, assign_optimized_random


def prepare_data(logger, dataset, feature_dir, label_path, window=None,
                 scaling='as_integer', smooth=False,
                 engineer_feature=None, new_bands_name=['ndvi'],
                 way='weekly', fill_missing='forward', check_missing=False,
                 vis_stack=False, vis_profile=False):
    """
    A pipeline to prepare data. The full process includes:
    - load raw bands
    - handle missing values
    - smooth bands or not (optional)
    - visualize timestamps (optional)
    - add new bands (optional)
    - engineer features
    - load labels (raw label / gt_cropland / gt_apples), and other auxiliary data (coordinates)
    - visualize profiles of new bands (optional)
    - get data with labels, discard unlabeled data (optional)
    - scale input data (optional)

    Parameters
    ----------
    logger
    dataset: string
        Indicate which dataset is under usage.
        choices = ['train_val', 'predict', f'test_{district}'].
    feature_dir: string
        Path to read the raster satellite images.
    label_path: string
        Path to read cleaned and labeled polygons.
    window: None or rasterio.window.Window
        The window to read *.tiff images.
    scaling: string
        Name of how to scale data.
        choices = ['as_float', 'as_reflectance', 'standardize', 'normalize']
    engineer_feature: choices = [None, 'temporal', 'temporal+spatial', 'select']
        Indicator of whether engineer features.
    new_bands_name: list
        A list of string about the name of newly added bands.
        choices = ['ndvi', 'ndre', 'gndvi', 'evi', 'cvi']
    way: string
        The way to stack raw timestamps.
    fill_missing: string
        How to fill missing values.
        choices = [None, 'forward', 'linear']
    vis_stack: bool
        Visualize timestamps or not.
    vis_profile: bool
        Visualize profiles of new bands or not.

    Returns
    -------

    """
    logger.info('### Prepare data')

    # load raw bands
    logger.info(f'# Stack timestamps {way}')
    if not isinstance(window, Window):
        meta, window, descriptions = prepare_meta_window_descriptions(feature_dir, label_path)
    else:
        meta, descriptions = prepare_meta_descriptions(feature_dir, window)
    read_as = 'as_integer' if 'as' not in scaling else scaling
    bands_array, cat_pixel, meta, timestamps_raw, timestamps_weekly_ref = \
        stack_timestamps(logger, feature_dir, meta, descriptions, window, read_as=read_as,
                         way=way, check_missing=check_missing)

    bands_name = list(descriptions)
    bands_name.remove('cloud mask')
    if fill_missing:
        logger.info(f"# Handle missing data by {fill_missing} filling")
        bands_array = handle_missing_data(bands_array, fill_missing, missing_val=0)
    if smooth:
        logger.info('# Smooth raw bands')
        bands_array = smooth_raw_bands(bands_array)
    if vis_stack:
        logger.info('# Visualize timestamp stacking')
        plot_timestamps(timestamps_raw, None, f'./figs/timestamps_raw_{dataset}.png')
        plot_timestamps(timestamps_weekly_ref, None, f'./figs/timestamps_{way}_{dataset}.png')

    logger.info('# Build features')
    if new_bands_name:
        bands_array = add_bands(logger, bands_array, descriptions, new_bands_name)
        bands_name += new_bands_name
        meta.update(count=meta['count'] - 1 + len(new_bands_name))
    df = build_features(logger, bands_array, engineer_feature, bands_name=bands_name)
    feature_names = df.columns
    logger.info(f'\nFeatures: {feature_names}')
    df['cat_mask'] = cat_pixel

    if 'predict' not in dataset:
        # get y
        logger.info('# Load raw labels')
        polygons_list, val_list, labels = load_shp_to_array(label_path, meta)
        df['label'] = labels.reshape(-1)
        logger.info('# Convert to cropland and crop labels')
        df['gt_cropland'] = df.label.values.copy()
        df.loc[df.label.values == 1, 'gt_cropland'] = 2
        df['gt_apples'] = df.label.values.copy()

        # add coordinates
        logger.info('# Add coordinates')
        df['coords'] = construct_coords(meta)

        # visualize profile weekly
        if vis_profile:
            for b in new_bands_name:
                type = 'cropland'
                b_arr = bands_array[:, :, bands_name.index(b), :]
                if smooth:
                    name = f"{b.upper()}_smoothed_{way}_profile_{dataset}_{type}"
                else:
                    name = f"{b.upper()}_{way}_profile_{dataset}_cropland"
                    name_s = f"{b.upper()}_smoothed_{way}_profile_{dataset}_{type}"
                    b_arr_smoothed = smooth_raw_bands(np.expand_dims(b_arr.copy(), axis=2))
                    plot_profile(data=b_arr_smoothed.reshape(-1, len(timestamps_weekly_ref)),
                                 label=df.label.values, timestamps=timestamps_weekly_ref, type=type,
                                 veg_index=b, title=name_s.replace('_', ' '), save_path=f"./figs/{name_s}.png")
                plot_profile(data=b_arr.reshape(-1, len(timestamps_weekly_ref)),
                             label=df.label.values, timestamps=timestamps_weekly_ref, type=type,
                             veg_index=b, title=name.replace('_', ' '), save_path=f"./figs/{name}.png")
        logger.info('ok')
        return df, meta, feature_names, polygons_list, val_list
    else:  
        logger.info('ok')
        return df, meta, feature_names


def handle_missing_data(arr, fill_missing, missing_val=0):
    """

    Parameters
    ----------
    arr: np.array
        shape(height, width, n_bands, n_weeks)
    fill_missing: string
    missing_val

    Returns
    -------
    arr: np.array
        shape(height, width, n_bands, n_weeks)
    """
    shape = arr.shape
    if fill_missing == 'forward':
        arr = forward_filling(arr.reshape(-1, shape[-1]), missing_val)
    elif fill_missing == 'linear':
        arr = linear_interpolation(arr.reshape(-1, shape[-1]), missing_val)
    return arr.reshape(shape)


def forward_filling(arr, missing_val=0):
    """
    Fill missing value using its previous one.

    Parameters
    ----------
    arr: np.array
        shape (height * width * n_bands, n_weeks)
    missing_val
        choices = [0, np.nan]

    Returns
    -------

    """
    mask = get_missing_values_mask(arr, missing_val)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    arr[mask] = arr[np.nonzero(mask)[0], idx[mask]]
    return arr


def linear_interpolation(arr, missing_val=0):
    """
    Fill missing value using the values before and after it.

    Parameters
    ----------
    arr: np.array
        shape (height * width * n_bands, n_weeks)
    missing_val: [0, np.nan]

    Returns
    -------

    """
    mask_arr = get_missing_values_mask(arr, missing_val)
    for i in np.nonzero(mask_arr.sum(axis=1))[0]:
        y, mask = arr[i, :], mask_arr[i, :]
        if mask.sum() < mask.shape[0] - 1:
            x = np.arange(y.shape[0])
            f = interpolate.interp1d(x[~mask], y[~mask], fill_value='extrapolate')
            arr[i, mask] = f(x[mask])
        elif mask.sum() == mask.shape[0] - 1:
            arr[i, :] = y[~mask]
        arr[i, arr[i] < 0] = 0
    return arr


def get_missing_values_mask(arr, missing_val):
    if missing_val == 0:
        mask = arr == 0.0
    elif missing_val == np.nan:
        mask = np.isnan(arr)
    else:
        raise ValueError(f'No such missing value indicator. Choose from [0, np.nan]')
    return mask


def smooth_raw_bands(bands_array):
    height, width, n_bands, n_weeks = bands_array.shape
    for h in range(height):
        for w in range(width):
            for b in range(n_bands):
                bands_array[h, w, b, :] = smooth_func(bands_array[h, w, b, :], 10)
    return bands_array


def smooth_func(y, box_pts=10):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def build_features(logger, bands_array, engineer_feature, bands_name):
    """

    Parameters
    ----------
    logger
    bands_array: nd.array
        shape (height, width, n_bands, n_weeks)
    engineer_feature: choices
        How to do feature engineering.
    bands_name: string
        Name of all bands we use, including derived new bands.

    Returns
    -------

    """
    # TODO: add spatial-related features, other low-resolution NIR bands
    height, width, n_bands, n_weeks = bands_array.shape
    bands_array = bands_array.reshape(height * width, n_bands, n_weeks)
    if engineer_feature:
        df = get_raw_every_n_weeks(logger, bands_name, n_weeks, bands_array, n=4)
        df_list = list()
        df_list.append(df)
        # statistics
        df_list.append(get_statistics(logger, bands_name, bands_array))
        # difference of two successive timestamps
        df_list.append(get_difference(logger, bands_name[4:], n_weeks, bands_array[:, 4:, :]))
        if engineer_feature == 'select' or engineer_feature == 'temporal+spatial':
            bands_array = bands_array.reshape(height, width, n_bands, n_weeks)
            df_list.append(get_all_spatial_features(logger, bands_name, bands_array))
        # concatenate
        df = pd.concat(df_list, axis=1)
    else:  # engineer_feature = None
        df = get_raw_every_n_weeks(logger, bands_name, n_weeks, bands_array, n=2)
    logger.info(f'  df.shape={df.shape}')
    return df


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


def construct_coords(meta):
    height, width = meta['height'], meta['width']
    minx, maxy = rasterio.transform.xy(meta['transform'], 0, 0)  # (691375.0, 3571325.0)
    maxx, miny = rasterio.transform.xy(meta['transform'], height, width)  # (709645.0, 3544635.0)
    xs = np.tile(np.linspace(minx, maxx, width, endpoint=False), reps=height)
    ys = np.linspace(maxy, miny, height, endpoint=False).repeat(width)
    xys = gpd.GeoSeries(gpd.points_from_xy(xs, ys))
    return xys


def get_valid_cropland_x_y(logger, df, n_feature, dataset):
    mask_valid = df.gt_cropland.values != 0
    df_valid = df[mask_valid]  # .reset_index(drop=True)
    x_valid = df_valid.iloc[:, :n_feature].values
    y_valid = df_valid.loc[:, 'gt_cropland'].values
    logger.info(
        f'df_{dataset}.shape {df_valid.shape}, x_{dataset}.shape {x_valid.shape}, y_{dataset}.shape {y_valid.shape}')
    logger.info(f'y_{dataset} with 3 classes:')
    count_classes(logger, df_valid.label.values)
    logger.info(f'y_{dataset} with 2 classes:')
    count_classes(logger, df_valid.gt_cropland.values)
    return df_valid, x_valid, y_valid


def get_valid_crop_x_y(logger, df, n_feature, dataset):
    mask_valid = (df.gt_apples.values == 1 or df.gt_apples.values == 2)
    df_valid = df[mask_valid]
    x_valid = df_valid.iloc[:, :n_feature].values
    y_valid = df_valid.loc[:, 'gt_apples'].values
    logger.info(
        f'df_{dataset}.shape {df_valid.shape}, x_{dataset}.shape {x_valid.shape}, y_{dataset}.shape {y_valid.shape}')
    logger.info(f'y_{dataset} with 2 classes:')
    count_classes(logger, df_valid.gt_apples.values)
    return df_valid, x_valid, y_valid


def clean_train_shapefiles(save_to_path='./data/train_labels/train_labels.shp'):
    # read all the shape files
    print(os.path.abspath(os.getcwd()))
    old_apples_shp = gpd.read_file('./data/apples/survey20210716_polygons20210819_corrected20210831.shp')
    new_apples_shp = gpd.read_file('./data/apples/survey20210825_polygons20210901_revised20210929.shp')
    non_crops_shp = gpd.read_file('./data/non_crops/non_crops.shp')
    other_crops_shp = gpd.read_file('./data/other_crops/other_crops.shp')
    train_region_shp = gpd.read_file('./data/train_region/train_region.shp')
    # put all shape files into one geo dataframe
    label_shp = gpd.GeoDataFrame(
        pd.concat([old_apples_shp, new_apples_shp, other_crops_shp, non_crops_shp], axis=0))
    if old_apples_shp.crs == new_apples_shp.crs == non_crops_shp.crs == other_crops_shp.crs:
        label_shp = label_shp.set_crs(new_apples_shp.crs)
    else:
        raise ValueError('crs of multiple files do not match.')
    # delete empty polygons and split multipolygons
    label_shp = label_shp.dropna().reset_index(drop=True)
    label_shp = multipolygons_to_polygons(label_shp)
    # mask for the study area
    save_label_in_region(label_shp, train_region_shp, save_to_path)


def clean_test_shapefiles():
    clean_test_near_shapefiles()
    clean_test_far_shapefiles()


def clean_test_near_shapefiles(save_to_path='./data/test_labels_kullu/test_labels_kullu.shp'):
    label_shp = gpd.read_file('./data/test_polygons_near/test_polygons_near.shp')
    test_region_shp = gpd.read_file('./data/test_region_near/test_region_near.shp')
    # delete empty polygons and split multipolygons
    label_shp = label_shp.dropna().reset_index(drop=True)
    label_shp = multipolygons_to_polygons(label_shp)
    # mask for the study area
    save_label_in_region(label_shp, test_region_shp, save_to_path)


def clean_test_far_shapefiles():
    old_apples_shp = gpd.read_file('./data/apples/survey20210716_polygons20210819_corrected20210831.shp')
    new_apples_shp = gpd.read_file('./data/apples/survey20210825_polygons20210901_revised20210929.shp')
    far_shp = gpd.read_file('./data/test_polygons_far/test_polygons_far.shp')  # include other crops and non-cropland
    label_shp = gpd.GeoDataFrame(
        pd.concat([old_apples_shp, new_apples_shp, far_shp], axis=0))
    if old_apples_shp.crs == new_apples_shp.crs == far_shp.crs:
        label_shp = label_shp.set_crs(new_apples_shp.crs)
    else:
        raise ValueError('crs of multiple files do not match.')
    # delete empty polygons and split multipolygons
    label_shp = label_shp.dropna().reset_index(drop=True)
    label_shp = multipolygons_to_polygons(label_shp)
    # check whether the dir exists
    mandi_path = './data/test_labels_mandi/test_labels_mandi.shp'
    shimla_path = './data/test_labels_shimla/test_labels_shimla.shp'
    if not os.path.exists(mandi_path.rstrip(mandi_path.split('/')[-1])):
        os.makedirs(mandi_path.rstrip(mandi_path.split('/')[-1]))
    if not os.path.exists(shimla_path.rstrip(shimla_path.split('/')[-1])):
        os.makedirs(shimla_path.rstrip(shimla_path.split('/')[-1]))
    label_shp[label_shp.district.values == 'Mandi'].to_file(mandi_path)
    label_shp[label_shp.district.values == 'Shimla'].to_file(shimla_path)


def save_label_in_region(label_shp, region_shp, save_to_path):
    if label_shp.crs != region_shp.crs:
        if label_shp.crs is None:
            label_shp = label_shp.to_crs(region_shp.crs)
        else:  # region_shp.crs == None
            region_shp = region_shp.to_crs(label_shp.crs)
    label_in_region = gpd.overlay(label_shp, region_shp, how='intersection')
    cols2drop = [col for col in ['id', 'id_2'] if col in label_in_region.columns]
    label_in_region = label_in_region.drop(cols2drop, axis=1).rename(columns={'id_1': 'id'})
    # check whether the dir exists
    folder_dir = save_to_path.rstrip(save_to_path.split('/')[-1])
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    label_in_region.to_file(save_to_path)  # save to folder


def get_unlabeled_data(df, unlabeled_mask, n_feature):
    x_unl = df.iloc[unlabeled_mask, :n_feature].values
    y_unl = np.zeros(x_unl.shape[0], dtype=int)
    grid_idx_unl = df.loc[unlabeled_mask, 'grid_idx'].values
    return x_unl, y_unl, grid_idx_unl
