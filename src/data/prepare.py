import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import interpolate
import pyproj
import shapely
import rasterio
from rasterio.windows import Window
from scipy.sparse import data

from src.data.load import load_geotiff
from src.data.stack import stack_timestamps
from src.data.engineer import add_vegetation_indices, get_temporal_features_every_n_weeks, get_statistical_features, \
    get_diff_features, get_spatial_features
from src.utils.util import count_classes, load_shp_to_array
from src.evaluation.visualize import plot_timestamps, plot_profile


def prepare_data(logger, dataset, feature_dir, label_path, window=None,
                 scaling='as_integer', smooth=False,
                 engineer_feature=None, new_bands_name=['ndvi'],
                 way='weekly', fill_missing='forward', check_missing=False, composite_by='max',
                 vis_stack=False, vis_profile=False, vis_profile_type='cropland', vis_afterprocess=False):
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
    label_path: string or None
        Path to read cleaned and labeled polygons.
    window: None or rasterio.window.Window
        The window to read *.tiff images.
    scaling: string
        Name of how to scale data.
        choices = ['as_float', 'as_reflectance', 'standardize', 'normalize']
    smooth: bool
        Smooth bands or not.
    engineer_feature: choices = [None, 'temporal', 'temporal+ndvi_spatial', 'temporal+spatial', 'select']
        Indicator of whether engineer features.
    new_bands_name: list
        A list of string about the name of newly added bands.
        choices = A list of any combination in ['ndvi', 'gndvi', 'evi', 'cvi']
    way: string
        The way to stack raw timestamps.
    fill_missing: string
        How to fill missing values.
        choices = [None, 'forward', 'linear']
    check_missing: bool
        Check missing value percentage or not.
    composite_by: str
        choices = [max, median]
    vis_stack: bool
        Visualize timestamps or not.
    vis_profile: bool
        Visualize profiles of new bands or not.
    vis_profile_type: choices = [cropland, apples]
    Returns
    -------

    """
    logger.info('### Prepare data')

    # load raw bands
    logger.info(f'# Stack timestamps {way}')
    if not isinstance(window, Window):
        meta, window, descriptions = get_meta_window_descriptions(
            feature_dir, label_path)
    else:
        meta, descriptions = get_meta_descriptions(feature_dir, window)

    
    df, bands_array, meta, feature_names, bands_name, timestamps_weekly_ref = \
        prepare_x(logger, dataset, feature_dir, window, meta, descriptions,
                  scaling=scaling, smooth=smooth,
                  engineer_feature=engineer_feature, new_bands_name=new_bands_name,
                  way=way, fill_missing=fill_missing, check_missing=check_missing,
                  vis_stack=vis_stack, composite_by=composite_by)
    
    
    
    if vis_afterprocess:
        meta_out = meta.copy()
        meta_out.update({'dtype': rasterio.float32})  # TODO: check update 
        for w in range(len(timestamps_weekly_ref)):
            save_path = './figs/after_process/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with rasterio.open(save_path + 'week_' + str(w) + '.tiff', 'w', **meta_out) as f:
                f.write(np.transpose(bands_array[:, :, :, w], (2, 0, 1)))
        logger.info(f'Saved images after process to {save_path}')

    if 'predict' not in dataset:
        # get y
        logger.info('# Load raw labels')
        polygons_list, labels = load_shp_to_array(label_path, meta)
        df['label'] = labels.reshape(-1)
        logger.info('# Convert to cropland and crop labels')
        df['gt_cropland'] = df.label.values.copy()
        df.loc[df.label.values == 1, 'gt_cropland'] = 2

        # add coordinates
        logger.info('# Add coordinates')
        df['coords'] = construct_coords(meta)

        # visualize profile weekly
        if vis_profile:
            for b in new_bands_name:
                b_arr = bands_array[:, :, bands_name.index(b), :]
                if smooth:
                    name = f"{b.upper()}_smoothed_{way}_profile_{dataset}_{vis_profile_type}"
                else:
                    name = f"{b.upper()}_{way}_profile_{dataset}_cropland"
                    name_s = f"{b.upper()}_smoothed_{way}_profile_{dataset}_{vis_profile_type}"
                    b_arr_smoothed = smooth_raw_bands(
                        np.expand_dims(b_arr.copy(), axis=2))
                    plot_profile(data=b_arr_smoothed.reshape(-1, len(timestamps_weekly_ref)),
                                 label=df.label.values, timestamps=timestamps_weekly_ref, type=vis_profile_type,
                                 veg_index=b, title=name_s.replace('_', ' '), save_path=f"./figs/{name_s}.png")
                plot_profile(data=b_arr.reshape(-1, len(timestamps_weekly_ref)),
                             label=df.label.values, timestamps=timestamps_weekly_ref, type=vis_profile_type,
                             veg_index=b, title=name.replace('_', ' '), save_path=f"./figs/{name}.png")
        logger.info('ok')
        return df, meta, feature_names, polygons_list
    else:
        logger.info('ok')
        return df, meta, feature_names


def prepare_HP_data(logger, img_dir, sample_HP_path,
                    smooth, engineer_feature, scaling, new_bands_name,
                    fill_missing, check_missing, composite_by):
    sample = gpd.read_file(sample_HP_path)
    # sample_shp = sample_shp.to_crs(meta.crs)

    df_list = list()
    for _, row in sample.iterrows():
        label, tile_id, poly = row['id'], row['tile'], row['geometry']
        feature_dir = img_dir + tile_id + '/raster/'
        with rasterio.open(feature_dir + os.listdir(feature_dir)[0], 'r') as f:
            meta_tile = f.meta
        # TODO: meta_tile and poly.bounds inconsistency
        # TODO: poly.buffer()
        project = pyproj.Transformer.from_proj(sample.crs, meta_tile['crs'], always_xy=True)
        poly = shapely.ops.transform(project.transform, poly)
        window = ow(meta_tile['transform'], poly.bounds)
        meta, descriptions = get_meta_descriptions(feature_dir, window)

        df_tile, _, meta, _, _, _ = \
            prepare_x(logger, 'train_val', feature_dir, window, meta, descriptions,
                      scaling=scaling, smooth=smooth,
                      engineer_feature=engineer_feature, new_bands_name=new_bands_name,
                      fill_missing=fill_missing, check_missing=check_missing, composite_by=composite_by)

        logger.info('# Load raw labels')
        df_tile['label'] = label
        logger.info('# Convert to cropland and crop labels')
        df_tile['gt_cropland'] = df_tile.label.values.copy()
        df_tile.loc[df_tile.label.values == 1, 'gt_cropland'] = 2

        # add coordinates
        logger.info('# Add coordinates')
        df_tile['coords'] = construct_coords(meta)

        # Add
        df_list.append(df_tile)
    df = pd.concat(df_list, axis=0)

    val_list = list(sample.id.values)
    features_list = [shapely.geometry.mapping(s) for s in sample.geometry]

    return df, features_list, val_list


def prepare_x(logger, dataset, feature_dir, window, meta, descriptions,
              scaling='as_integer', smooth=False,
              engineer_feature=None, new_bands_name=['ndvi'],
              way='weekly', fill_missing='forward', check_missing=False,
              vis_stack=False, composite_by='max'):
    read_as = 'as_integer' if 'as' not in scaling else scaling
    bands_array, cat_pixel, meta, timestamps_raw, timestamps_weekly_ref = \
        stack_timestamps(logger, feature_dir, meta, descriptions, window, read_as=read_as,
                         way=way, check_missing=check_missing, composite_by=composite_by)

    bands_name = list(descriptions)
    bands_name.remove('cloud mask')
    if fill_missing:
        # TODO: fill according to --predict_train to save computational cost
        logger.info(f"# Handle missing data by {fill_missing} filling")
        bands_array = handle_missing_data(
            bands_array, fill_missing, missing_val=0)
    if smooth:
        logger.info('# Smooth raw bands')
        bands_array = smooth_raw_bands(bands_array)
    if vis_stack:
        logger.info('# Visualize timestamp stacking')
        plot_timestamps(timestamps_raw, None,
                        f'./figs/timestamps_raw_{dataset}.png')
        plot_timestamps(timestamps_weekly_ref, None,
                        f'./figs/timestamps_{way}_{dataset}.png')

    logger.info('# Build features')
    if new_bands_name:
        bands_array = add_vegetation_indices(
            logger, bands_array, descriptions, new_bands_name)
        bands_name += new_bands_name
        meta.update(count=meta['count'] + len(new_bands_name))
    df = build_features(logger, bands_array,
                        engineer_feature, bands_name=bands_name)
    feature_names = df.columns
    logger.info(f'\nFeatures: {feature_names}')
    df['cat_mask'] = cat_pixel

    return df, bands_array, meta, feature_names, bands_name, timestamps_weekly_ref


def handle_missing_data(arr, fill_missing, missing_val=0):
    """

    Parameters
    ----------
    arr: np.array
        shape(height, width, n_bands, n_weeks)
    fill_missing: choices = [forward, linear]
        Way to fill missing values.
    missing_val: choices = [float, int, np.nan]
        A token being recognized as missing.

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
    missing_val: choices = [float, int, np.nan]
        A token being recognized as missing.

    Returns
    -------
    arr: np.array
        shape(height * width * n_bands, n_weeks)
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
    missing_val: choices = [float, int, np.nan]

    Returns
    -------
    arr: np.array
        shape(height * width * n_bands, n_weeks)
    """
    mask_arr = get_missing_values_mask(arr, missing_val)
    for i in np.nonzero(mask_arr.sum(axis=1))[0]:
        y, mask = arr[i, :], mask_arr[i, :]
        if mask.sum() < mask.shape[0] - 1:
            x = np.arange(y.shape[0])
            f = interpolate.interp1d(
                x[~mask], y[~mask], fill_value='extrapolate')
            arr[i, mask] = f(x[mask])
        elif mask.sum() == mask.shape[0] - 1:
            arr[i, :] = y[~mask]
        arr[i, arr[i] < 0] = 0
    return arr


def get_missing_values_mask(arr, missing_val):
    if isinstance(missing_val, float) or isinstance(missing_val, int):
        mask = arr == missing_val
    elif missing_val == np.nan:
        mask = np.isnan(arr)
    else:
        raise ValueError(
            f'No such missing value indicator. Choose from any float / any int / np.nan.')
    return mask


def smooth_raw_bands(bands_array):
    """
    Smooth raw bands pixel by pixel.

    Parameters
    ----------
    bands_array: ndarray
        shape (height, width, n_bands, n_weeks)

    Returns
    -------
    bands_array: ndarray
        shape (height, width, n_bands, n_weeks)
    """
    height, width, n_bands, _ = bands_array.shape
    for h in range(height):
        for w in range(width):
            for b in range(n_bands):
                bands_array[h, w, b, :] = smooth_func(
                    bands_array[h, w, b, :], 10)
    return bands_array


def smooth_func(y, box_pts=10):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def build_features(logger, bands_array, engineer_feature, bands_name):
    """
    Build features according to the engineer_feature type.

    Parameters
    ----------
    logger
    bands_array: nd.array
        shape (height, width, n_bands, n_weeks)
    engineer_feature: choices = [None, temporal, tempora+ndvi_spatial, tempora+spatial, select]
        How to do feature engineering.
    bands_name: list
        A list of bands' name used, including derived new bands.

    Returns
    -------
    df: pd.DataFrame
        shape (n_pixels, n_feature) where n_pixels = height * width

    """
    # TODO: add spatial-related features, other low-resolution NIR bands
    height, width, n_bands, n_weeks = bands_array.shape
    bands_array = bands_array.reshape(height * width, n_bands, n_weeks)
    if engineer_feature:
        df = get_temporal_features_every_n_weeks(
            logger, bands_name, n_weeks, bands_array, n=4)
        df_list = list()
        df_list.append(df)
        # statistics
        df_list.append(get_statistical_features(
            logger, bands_name, bands_array))
        # difference of two successive timestamps
        df_list.append(get_diff_features(
            logger, bands_name[4:], n_weeks, bands_array[:, 4:, :]))
        if engineer_feature == 'select' or engineer_feature == 'temporal+spatial':
            bands_array = bands_array.reshape(height, width, n_bands, n_weeks)
            df_list.append(get_spatial_features(
                logger, bands_name, bands_array))
        elif engineer_feature == 'temporal+ndvi_spatial':
            bands_array = bands_array.reshape(height, width, n_bands, n_weeks)
            df_list.append(get_spatial_features(
                logger, bands_name[4:], bands_array[:, :, 4:, :]))
        # concatenate
        df = pd.concat(df_list, axis=1)
    else:  # engineer_feature = None
        df = get_temporal_features_every_n_weeks(
            logger, bands_name, n_weeks, bands_array, n=1)
    logger.info(f'  df.shape={df.shape}')
    return df


def construct_coords(meta):
    height, width = meta['height'], meta['width']
    minx, maxy = rasterio.transform.xy(
        meta['transform'], 0, 0)  # (691375.0, 3571325.0)
    maxx, miny = rasterio.transform.xy(
        meta['transform'], height, width)  # (709645.0, 3544635.0)
    xs = np.tile(np.linspace(minx, maxx, width, endpoint=False), reps=height)
    ys = np.linspace(maxy, miny, height, endpoint=False).repeat(width)
    xys = gpd.GeoSeries(gpd.points_from_xy(xs, ys))
    return xys


def get_valid_cropland_x_y(logger, df, n_feature, dataset):
    """
    Get all the annotated data. (label is not 0)

    Parameters
    ----------
    logger
    df: pd.DataFrame
        shape (n_pixel, n_feature + other columns)
    n_feature: int
        The number of features.
    dataset: string
        Operate on which dataset.

    Returns
    -------
    df_valid: pd.DataFrame
        shape (n_valid, n_feature + other columns)
    x_valid: np.array
        shape (n_valid, n_feature)
    y_valid: np.array
        shape (n_valid, )
    """
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


def get_crop_type_x_y_pos(logger, df, n_feature, dataset):
    """
    Get only the croplands data.

    Parameters
    ----------
    logger
    df: pd.DataFrame
        shape (n_pixel, n_feature + other columns)
    n_feature: int
        The number of features.
    dataset: string
        Operate on which dataset.

    Returns
    -------
    df_valid: pd.DataFrame
        shape (n_valid, n_feature + other columns)
    x_valid: np.array
        shape (n_valid, n_feature)
    y_valid: np.array
        shape (n_valid, )
    """
    mask_valid = (df.label.values == 1)
    df_valid = df[mask_valid]
    x_valid = df_valid.iloc[:, :n_feature].values
    y_valid = df_valid.loc[:, 'label'].values
    logger.info(
        f'df_{dataset}.shape {df_valid.shape}, x_{dataset}.shape {x_valid.shape}, y_{dataset}.shape {y_valid.shape}')
    logger.info(f'y_{dataset} with 2 classes:')
    count_classes(logger, df_valid.label.values)
    return df_valid, x_valid, y_valid


def get_meta_window_descriptions(geotiff_dir, label_path):
    """
    Get meta / band descriptions from raster file, and window from labeled polygons.

    Parameters
    ----------
    geotiff_dir: string
        Path storing raster files.
    label_path: string
        Path of labeled polygons.

    Returns
    -------
    meta: dict
        Meta data of the window.
    window: rasterio.window.Window
        The window to load images.
    description: list
        A list of ordered band name.
    """
    img_path = geotiff_dir + os.listdir(geotiff_dir)[0]
    img = rasterio.open(img_path)
    transform, dst_crs, descriptions = img.transform, img.crs, img.descriptions

    label_shp = gpd.read_file(label_path)
    
    # Filtering for a distrixt
    label_shp = label_shp[label_shp.district=='Kullu']

    label_shp = label_shp.to_crs(dst_crs)
    

    window = get_window(transform, label_shp.total_bounds)

    _, meta = load_geotiff(img_path, window, read_as='as_integer')
    return meta, window, descriptions


def get_meta_descriptions(geotiff_dir, window):
    """
    Get meta / band descriptions from raster file, given defined window.

    Parameters
    ----------
    geotiff_dir: string
        Path storing raster files.
    window: rasterio.window.Window
        The window to load images.

    Returns
    -------
    meta: dict
        Meta data of the window.
    description: list
        A list of ordered band name.
    """
    img_path = geotiff_dir + os.listdir(geotiff_dir)[0]
    img = rasterio.open(img_path)

    _, meta = load_geotiff(img_path, window, read_as='as_integer')
    return meta, img.descriptions


def get_window(transform, bounds):
    """
    Get the window from bounds given transformation.

    Parameters
    ----------
    transform: rasterio.transform
        A transformation between rowcol and xy.
    bounds: tuple
        A tuple of box coordinates.

    Returns
    -------
    Window(col_off, row_off, width, height)
    """
    minx, miny, maxx, maxy = bounds
    row_min, col_min = rasterio.transform.rowcol(transform, minx, maxy)
    row_max, col_max = rasterio.transform.rowcol(transform, maxx, miny)
    return Window(col_min, row_min, col_max - col_min, row_max - row_min)
