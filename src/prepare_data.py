import os
import stat
import pyproj
import datetime
import math
import random
import fiona
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import skimage
import skimage.draw
import pyproj
from visualization import plot_timestamps
from util import stack_all_timestamps, count_classes, load_target_shp, compute_mask, multipolygons_to_polygons
from feature_engineering import add_bands, get_raw_monthly, get_statistics, get_difference
from sklearn.model_selection import train_test_split as random_train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def prepare_data(logger, dataset, feature_dir, label_path, vis_ts=False, feature_scaling=None,
                 scaler=None, way='weekly', interpolation='previous'):
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
    vis_ts
    feature_scaling
    scaler
    way
    interpolation

    Returns
    -------

    """
    logger.info('### Prepare data')

    # load raw bands
    logger.info(f'# Stack all timestamps {way}')
    bands_array, meta, timestamps_raw, timestamps_weekly, timestamps_weekly_ref = \
        stack_all_timestamps(logger, feature_dir, way=way, interpolation=interpolation)

    # visualize
    if vis_ts:
        logger.info('# Visualize timestamps')
        plot_timestamps(timestamps_raw, '../figs/timestamps_raw.png')
        plot_timestamps(timestamps_weekly_ref, '../figs/timestamps_weekly.png')

    # get x
    logger.info('# Engineer features')
    df = engineer_features(logger, bands_array, timestamps_weekly_ref,
                           new_bands_name=['ndvi'])
    feature_names = df.columns
    n_feature = feature_names.shape[0]

    # get y
    logger.info('# Load raw labels')
    labels = get_labels(label_path, meta)
    df['label'] = labels.reshape(-1)
    logger.info('# Convert to cropland labels')
    df['gt_cropland'] = df.label.values.copy()
    df.loc[df.label.values == 1, 'gt_cropland'] = 2

    # add coordinates
    logger.info('# Add coordinates')
    coords = construct_coordinates(meta['width'], meta['height'])
    df['coords'] = coords
    # TODO: AttributeError: 'Series' object has no attribute 'total_bounds' if df['coords'] = coords

    # get valid and whole
    df_valid, x_valid, y_valid, coords_valid = \
        get_valid_x_y(logger, df=df, coords=coords, n_feature=n_feature, dataset=dataset)

    # normalize
    if feature_scaling is not None:
        logger.info(f'# {feature_scaling} features')
        if dataset == 'train_val':
            if feature_scaling == 'normalize':
                scaler = MinMaxScaler().fit(x_valid)
            else:  # feature_scaling == 'standardize'
                scaler = StandardScaler().fit(x_valid)
        x_valid = scaler.transform(x_valid)

    logger.info('ok')

    return df, df_valid, x_valid, y_valid, coords_valid, scaler, meta, n_feature, feature_names


# TODO: distinguish feature engineering and raw feature by flag --feature_engineering
def engineer_features(logger, bands_array, timestamps_weekly_ref, new_bands_name=['ndvi']):
    bands_name = ['blue', 'green', 'red', 'nir']
    # add more features
    if new_bands_name is not None:
        bands_array = add_bands(logger, bands_array, new_bands_name)
        bands_name += new_bands_name
    num_of_weeks = len(timestamps_weekly_ref)

    # raw features
    df = get_raw_monthly(logger, bands_name, num_of_weeks, bands_array)
    df_list = list()
    df_list.append(df)
    # statistics
    df_list.append(get_statistics(logger, bands_name, num_of_weeks, bands_array))
    # difference of two successive timestamps
    df_list.append(get_difference(logger, new_bands_name, num_of_weeks, bands_array))
    # concatenate
    df = pd.concat(df_list, axis=1)
    logger.info(f'  df.shape={df.shape}')
    return df


def get_labels(label_path, meta):
    polygons, rc_polygons, class_list = \
        load_target_shp(label_path,
                        transform=meta['transform'],
                        proj_out=pyproj.Proj(meta['crs']))
    labels = compute_mask(rc_polygons, meta, class_list)
    return labels


def construct_coordinates(width, height):
    Ys = np.tile(np.arange(width), reps=height)
    Xs = np.tile(np.arange(height).reshape(-1, 1), reps=width).reshape(-1)
    XYs = gpd.GeoSeries(gpd.points_from_xy(Ys, Xs))
    return XYs


def get_valid_x_y(logger, df, coords, n_feature, dataset):
    mask_valid = df.gt_cropland.values != 0
    df_valid = df[mask_valid].reset_index(drop=True)
    coords_valid = coords[mask_valid].reset_index(drop=True)
    x_valid = df_valid.iloc[:, :n_feature].values
    y_valid = df_valid.loc[:, 'gt_cropland'].values
    logger.info(
        f'df_{dataset}.shape {df_valid.shape}, x_{dataset}.shape {x_valid.shape}, y_{dataset}.shape {y_valid.shape}')
    logger.info(f'y_{dataset} with 3 classes:')
    count_classes(logger, df_valid.label.values)
    logger.info(f'y_{dataset} with 2 classes:')
    count_classes(logger, df_valid.gt_cropland.values)
    return df_valid, x_valid, y_valid, coords_valid


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
