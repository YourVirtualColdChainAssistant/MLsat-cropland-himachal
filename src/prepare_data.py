import os
import stat
import pyproj
import datetime
import math
import random
import fiona
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import skimage
import skimage.draw
import pyproj
from visualization import plot_timestamps
from util import stack_all_timestamps, count_classes, load_target_shp, compute_mask, multipolygons_to_polygons
from feature_engineering import add_bands, get_raw_every_n_weeks, get_statistics, get_difference
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from spacv.utils import geometry_to_2d
from sklearn.neighbors import BallTree
from spacv.grid_builder import construct_grid, assign_randomized, assign_systematic, assign_optimized_random
from visualization import plot_smoothed_ndvi_profile


def prepare_data(logger, dataset, feature_dir, label_path,
                 feature_scaling=None, scaler=None, feature_engineering=True, new_bands_name=['ndvi'],
                 way='weekly', interpolation='previous',
                 vis_ts=False, vis_profile=False):
    """
    A pipeline to prepare data. The full process includes:
    1. load raw bands
    2. visualize timestamps (optional)
    3. engineer features, get x
    4. load labels, get y
    5. normalize x (optional)
    6. add coordinates

    Parameters
    ----------
    logger
    dataset
    feature_dir
    label_path
    feature_scaling
    feature_engineering
    new_bands_name
    scaler
    way
    interpolation
    vis_ts
    vis_profile

    Returns
    -------

    """
    logger.info('### Prepare data')

    # load raw bands
    logger.info(f'# Stack all timestamps {way}')
    bands_array, meta, timestamps_raw, timestamps_weekly, timestamps_weekly_ref = \
        stack_all_timestamps(logger, feature_dir, way=way, interpolation=interpolation)
    if feature_engineering:
        logger.info('# Smooth raw bands')
        bands_array = smooth_raw_bands(bands_array)

    # visualize
    if vis_ts:
        logger.info('# Visualize timestamps')
        plot_timestamps(timestamps_raw, f'../figs/timestamps_raw_{dataset}.png')
        plot_timestamps(timestamps_weekly_ref, f'../figs/timestamps_weekly_{dataset}.png')

    # get x
    logger.info('# Build features')
    bands_array, df, bands_name = build_features(logger, bands_array, feature_engineering,
                                                 timestamps_weekly_ref, new_bands_name=new_bands_name)
    feature_names = df.columns
    n_feature = feature_names.shape[0]

    # get y
    logger.info('# Load raw labels')
    polygons, labels = get_labels(label_path, meta)
    df['label'] = labels.reshape(-1)
    logger.info('# Convert to cropland labels')
    df['gt_cropland'] = df.label.values.copy()
    df.loc[df.label.values == 1, 'gt_cropland'] = 2

    # add coordinates
    logger.info('# Add coordinates')
    df['coords'] = construct_coords(meta)
    # construct_coords(meta).total_bounds: ok
    # df['coords'].values.total_bounds: ok
    # df['coords'].total_bounds: 'Series' object has no attribute 'total_bounds'
    # Maybe because df is DataFrame rather than GeoDataFrame?

    # get valid and whole
    df_valid, x_valid, y_valid = \
        get_valid_x_y(logger, df=df, n_feature=n_feature, dataset=dataset)

    # normalize
    if feature_scaling is not None:
        logger.info(f'# {feature_scaling} features')
        if dataset == 'train_val':
            if feature_scaling == 'normalize':
                scaler = MinMaxScaler().fit(x_valid)
            else:  # feature_scaling == 'standardize'
                scaler = StandardScaler().fit(x_valid)
        x_valid = scaler.transform(x_valid)

    if vis_profile:
        # visualize ndvi profile weekly
        ndvi_array = bands_array[:, bands_name.index('ndvi'), :].reshape(-1, len(timestamps_weekly_ref))
        plot_smoothed_ndvi_profile(ndvi_array, df.label.values, timestamps_weekly_ref,
                                   title=f'NDVI weekly profile smoothed ({dataset})',
                                   save_path=f'../figs/NDVI_weekly_smoothed_{dataset}.png')
        n_month = math.floor(len(timestamps_weekly_ref) / 4)
        ndvi_array_monthly = ndvi_array[..., :(4 * n_month)].reshape(ndvi_array.shape[0], n_month, 4).max(axis=2)
        plot_smoothed_ndvi_profile(ndvi_array_monthly, df.label.values, timestamps_weekly_ref[::4][:n_month],
                                   title=f'NDVI monthly profile smoothed ({dataset})',
                                   save_path=f'../figs/NDVI_monthly_smoothed_{dataset}.png')

    logger.info('ok')

    return df, df_valid, x_valid, y_valid, polygons, scaler, meta, n_feature, feature_names


def smooth_raw_bands(bands_array):
    n_pixels, n_bands, n_weeks = bands_array.shape
    for p in range(n_pixels):
        for b in range(n_bands):
            bands_array[p, b, :] = smooth(bands_array[p, b, :], 10)
    return bands_array


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def build_features(logger, bands_array, feature_engineering, timestamps_weekly_ref, new_bands_name=['ndvi']):
    """

    Parameters
    ----------
    logger
    bands_array: ndarray
        shape (n_pixels, n_bands, n_weeks)
    feature_engineering
    timestamps_weekly_ref
    new_bands_name

    Returns
    -------

    """
    bands_name = ['blue', 'green', 'red', 'nir']
    # add more features
    if new_bands_name is not None:
        bands_array = add_bands(logger, bands_array, new_bands_name)
        bands_name += new_bands_name
    n_weeks = len(timestamps_weekly_ref)

    if feature_engineering:
        df = get_raw_every_n_weeks(logger, bands_name, n_weeks, bands_array, n=4)
        df_list = list()
        df_list.append(df)
        # statistics
        df_list.append(get_statistics(logger, bands_name, bands_array))
        # difference of two successive timestamps
        df_list.append(get_difference(logger, new_bands_name, n_weeks, bands_array))
        # concatenate
        df = pd.concat(df_list, axis=1)
    else:  # feature_engineering = False
        df = get_raw_every_n_weeks(logger, bands_name, n_weeks, bands_array, n=2)
    logger.info(f'  df.shape={df.shape}')
    return bands_array, df, bands_name


def get_labels(label_path, meta):
    features, polygons, rc_polygons, class_list = \
        load_target_shp(label_path,
                        transform=meta['transform'],
                        proj_out=pyproj.Proj(meta['crs']))
    labels = compute_mask(rc_polygons, meta, class_list)
    return features, labels


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
    polygons_gpd = assign_polygons_to_grid(polygons_gpd, grid_all)  # columns=[geometry, grid_id]
    grid_valid = grid_all.loc[sorted(polygons_gpd.grid_id.unique().astype(int)), :].reset_index(drop=True)

    # for g_id in grid_valid.grid_id:
    #     grid_valid.loc[g_id, 'geometry'] = grid_valid.loc[g_id, 'geometry'].union(
    #         polygons_gpd.geometry[polygons_gpd.grid_id.values == g_id].unary_union)
    #     grid_valid.loc[grid_valid.grid_id.values != g_id, 'geometry'] = grid_valid.loc[
    #         grid_valid.grid_id.values != g_id, 'geometry'].difference(
    #         polygons_gpd.geometry[polygons_gpd.grid_id.values != g_id].unary_union)

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

    # update grid shape
    # for poly, grid_id in zip(polygons[border_polygon_index].geometry, polygons[border_polygon_index].grid_id):
    #     # !!Q 1) multipolygon produced by making difference,
    #     # 2) ValueError: Must have equal len keys and value when setting with an iterable
    #     grid.loc[grid_id, 'geometry'] = grid.loc[grid_id, 'geometry'].union(poly)
    #     grid.loc[grid.grid_id.values != grid_id, 'geometry'] = \
    #         grid.loc[grid.grid_id.values != grid_id, 'geometry'].difference(poly)
    return polygons


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
    pass


# def construct_fold(grid):
#     fold = gpd.GeoDataFrame(columns=['geometry'])
#     for f_id in grid.fold_id.unique():
#         grids = grid.geometry[grid.fold_id == f_id]
#         print(f_id, grids.shape)
#         fold.append({'geometry': shapely.ops.unary_union(grids)})
#     return foldstruct_fold(grid):
#     fold = gpd.GeoDataFrame(columns=['geometry'])
#     for f_id in grid.fold_id.unique():
#         grids = grid.geometry[grid.fold_id == f_id]
#         print(f_id, grids.shape)
#         fold.append({'geometry': shapely.ops.unary_union(grids)})
#     return fold


def construct_coords(meta):
    height, width = meta['height'], meta['width']
    minx, maxy = rasterio.transform.xy(meta['transform'], 0, 0)  # 77.03217968, 32.02331729
    maxx, miny = rasterio.transform.xy(meta['transform'], height, width)  # 77.2176813 , 32.25444632
    xs = np.tile(np.linspace(minx, maxx, width, endpoint=False), reps=height)
    ys = np.linspace(maxy, miny, height, endpoint=False).repeat(width)
    xys = gpd.GeoSeries(gpd.points_from_xy(xs, ys))
    return xys


def get_valid_x_y(logger, df, n_feature, dataset):
    mask_valid = df.gt_cropland.values != 0
    df_valid = df[mask_valid].reset_index(drop=True)
    x_valid = df_valid.iloc[:, :n_feature].values
    y_valid = df_valid.loc[:, 'gt_cropland'].values
    logger.info(
        f'df_{dataset}.shape {df_valid.shape}, x_{dataset}.shape {x_valid.shape}, y_{dataset}.shape {y_valid.shape}')
    logger.info(f'y_{dataset} with 3 classes:')
    count_classes(logger, df_valid.label.values)
    logger.info(f'y_{dataset} with 2 classes:')
    count_classes(logger, df_valid.gt_cropland.values)
    return df_valid, x_valid, y_valid


def clean_train_shapefiles(save_to_path='../data/train_labels/train_labels.shp'):
    # read all the shape files
    old_apples_shp = gpd.read_file('../data/apples/survey20210716_polygons20210819_corrected20210831.shp')
    new_apples_shp = gpd.read_file('../data/apples/survey20210825_polygons20210901_revised20210929.shp')
    non_crops_shp = gpd.read_file('../data/non_crops/non_crops.shp')
    other_crops_shp = gpd.read_file('../data/other_crops/other_crops.shp')
    train_area_shp = gpd.read_file('../data/train_area/train_area.shp')
    # put all shape files into one geo dataframe
    label_shp = gpd.GeoDataFrame(
        pd.concat([old_apples_shp, new_apples_shp, other_crops_shp, non_crops_shp], axis=0))
    # delete empty polygons
    label_shp = label_shp.dropna().reset_index(drop=True)
    # split multipolygons
    label_shp = multipolygons_to_polygons(label_shp)
    # mask for the study area
    get_label_in_region(label_shp, train_area_shp, save_to_path)


def clean_test_shapefiles(save_to_path='../data/test_labels/test_labels.shp'):
    label_shp = gpd.read_file('../data/test_polygons/test_polygons.shp')
    test_region_shp = gpd.read_file('../data/test_region_near/test_region_near.shp')
    get_label_in_region(label_shp, test_region_shp, save_to_path)


def get_label_in_region(label_shp, region_shp, save_to_path):
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
