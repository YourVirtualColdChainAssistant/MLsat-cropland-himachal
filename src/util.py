import os
import re
import sys
import fiona
import logging
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import skimage
import skimage.draw
import pyproj
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn.inspection import permutation_importance


def load_geotiff(path, window=None):
    """ Load the geotiff as a list of numpy array.
        INPUT : path (str) -> the path to the geotiff
                window (rasterio.windows.Window) -> the window to use when loading the image
        OUTPUT : band (list of numpy array) -> the different bands as float scaled to 0:1
                 meta (dictionary) -> the metadata associated with the geotiff
    """
    with rasterio.open(path) as f:
        band = [skimage.img_as_float(f.read(i+1, window=window)) for i in range(f.count)]
        meta = f.meta
        if window is not None:
            meta['height'] = window.height
            meta['width'] = window.width
            meta['transform'] = f.window_transform(window)
    return band, meta


def upsampling_20m_to_10m():
    pass


def split_raster():
    pass


def clip_raster(images_dir, clip_from_shp='../data/study-area/study_area.shp'):
    # shape file information
    with fiona.open(clip_from_shp, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile if feature["geometry"] is not None]
    shape_crs = gpd.read_file(clip_from_shp).crs

    # geotiff directory
    geotiff_dir = images_dir + 'geotiff/'
    # clip directory
    clip_dir = images_dir + 'clip/'
    if not os.path.exists(clip_dir):
        os.mkdir(clip_dir)

    # clip all the raster
    filenames = [f for f in sorted(os.listdir(geotiff_dir)) if f.endswith('.tiff')]
    print('\nStart clipping...')
    for i, filename in enumerate(filenames, start=1):
        geotiff_filepath = geotiff_dir + filename
        clip_filepath = clip_dir + filename
        if not is_clipped(clip_filepath):
            print(f'[{i}/{len(filenames)}] Clipping {clip_filepath}')
            clip_single_raster(shape_crs, shapes, geotiff_filepath, clip_filepath)
        else:
            print(f'[{i}/{len(filenames)}] {clip_filepath} clipped')
    print(f'Clip done!')


def is_clipped(clip_filepath):
    if os.path.exists(clip_filepath) and not os.path.exists(clip_filepath + '.aux.xml'):
        return True
    else:
        return False


def clip_single_raster(shape_crs, shapes, geotiff_filepath, clip_filepath):
    # get the coordinate system of raster
    raster = rasterio.open(geotiff_filepath)

    # check if two coordinate systems are the same
    if shape_crs != raster.crs:
        reproject_single_raster(shape_crs, geotiff_filepath, clip_filepath)
        # read imagery file
        with rasterio.open(clip_filepath) as src:
            out_image, out_transform = mask(src, shapes, crop=True)
            out_meta = src.meta
    else:
        # read imagery file
        with rasterio.open(geotiff_filepath) as src:
            out_image, out_transform = mask(src, shapes, crop=True)
            out_meta = src.meta

    # Save clipped imagery
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(clip_filepath, "w", **out_meta) as dst:
        # out_image.shape (band, height, width)
        dst.write(out_image)


def reproject_single_raster(dst_crs, input_file, transformed_file):
    """
    :param dst_crs: output projection system
    :param input_file
    :param transformed_file
    :return:
    """
    with rasterio.open(input_file) as imagery:
        transform, width, height = calculate_default_transform(imagery.crs, dst_crs, imagery.width, imagery.height,
                                                               *imagery.bounds)
        kwargs = imagery.meta.copy()
        kwargs.update({'crs': dst_crs, 'transform': transform, 'width': width, 'height': height})
        with rasterio.open(transformed_file, 'w', **kwargs) as dst:
            for i in range(1, imagery.count + 1):
                reproject(
                    source=rasterio.band(imagery, i),
                    destination=rasterio.band(dst, i),
                    src_transform=imagery.transform,
                    src_crs=imagery.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


def timestamp_sanity_check(timestamp_std, filename):
    timestamp_get = datetime.datetime.strptime(re.split('[_.]', filename)[-2],
                                      '%Y%m%dT%H%M%S%f').date()
    timestamp_get = timestamp_get - datetime.timedelta(days=timestamp_get.weekday())
    if timestamp_std != timestamp_get:
        print(f'{timestamp_std} and {timestamp_get} do not match!')
        exit()


def choices_sanity_check(choices, choice, var_name):
    if choice not in choices:
        print(f'{choice} is unavailable. Please choose "{var_name}" from {choices}')
        exit()


def get_weekly_timestamps():
    """
    Get the date of all Monday in 2020.

    """
    date_start = datetime.date(2020, 1, 1)
    date_end = datetime.date(2020, 12, 31)
    date_start = date_start - datetime.timedelta(days=date_start.weekday())
    date_end = date_end - datetime.timedelta(days=date_end.weekday())
    d = date_start
    weekly_timestamps = []
    while d <= date_end:
        weekly_timestamps.append(d)
        d += datetime.timedelta(7)
    return weekly_timestamps


def get_monthly_timestamps():
    """
    Get the first day of each month in 2020.

    """
    date_start = datetime.date(2020, 1, 1)
    monthly_timestamps = []
    for m in range(1, 13):
        monthly_timestamps.append(date_start.replace(month=m))
    return monthly_timestamps


def stack_all_timestamps(logger, from_dir, way='weekly', interpolation='previous'):
    """
    Stack all the timestamps in from_dir folder, ignoring all black images.

    :param logger: std::out
    :param from_dir: string
    :param way: string
        choices = ['raw', 'weekly', 'monthly']
    :param interpolation: string
        choices = ['zero', 'previous']

    :return: bands_array, meta, timestamps_bf, timestamps_af, timestamps_weekly
    bands_array: array
        shape (pixel, number of bands, number of weeks)
    timestamps_bf: the raw timestamps
    """
    logger.info(f'--- Stacking all timestamps {way} ---')

    # ### sanity check
    choices_sanity_check(['raw', 'weekly', 'monthly'], way, 'way')
    choices_sanity_check(['zero', 'previous'], interpolation, 'interpolation')

    # ### raw files
    # sorted available files
    filenames = sorted(os.listdir(from_dir))
    # get bands' meta data
    _, meta = load_geotiff(from_dir + os.listdir(from_dir)[0])
    # find all the raw time stamps
    timestamps_bf = []
    for filename in filenames:
        timestamps_bf.append(datetime.datetime.strptime(re.split('[_.]', filename)[-2],
                                               '%Y%m%dT%H%M%S%f').date())

    # ### check the way to stack
    if way == 'raw':
        timestamps_af = timestamps_bf
        timestamps_ref = timestamps_bf
    elif way == 'weekly':
        timestamps_af = [ts - datetime.timedelta(days=ts.weekday()) for ts in
                         timestamps_bf]  # datetime.weekday() returns 0~6
        timestamps_ref = get_weekly_timestamps()
    else:
        timestamps_af = [ts - datetime.timedelta(days=ts.day-1) for ts in timestamps_bf]  # datetime.day returns 1-31
        timestamps_ref = get_monthly_timestamps()

    # ### stack all the timestamps
    timestamps_af_pd = pd.Series(timestamps_af)
    bands_list, black_ids = [], []
    for i, timestamp in enumerate(timestamps_ref, start=1):
        # get all the indices
        ids = timestamps_af_pd[timestamps_af_pd.eq(timestamp)].index
        band_list = []
        # with non-empty data, check missing data
        if len(ids) != 0:
            for id in ids:
                # read band
                filename = filenames[id]
                raster_filepath = from_dir + filename
                band, meta = load_geotiff(raster_filepath)
                # sanity check
                timestamp_sanity_check(timestamp, raster_filepath)
                # pixel values check
                if np.array(band).mean() != 0.0:
                    band_list.append(np.stack(band, axis=2).reshape(-1, len(band)))
                else:
                    black_ids.append(id)
                    logger.info(f'  Discard {filename} due to empty value.')
        # stack by index
        if len(band_list) != 0:
            band_list = np.stack(band_list, axis=2).mean(axis=2)
            # format printing
            print_str = ''
            for id in ids:
                if id in black_ids:
                    print_str += f'x{timestamps_bf[id].strftime("%Y-%m-%d")}, '
                else:
                    print_str += f'{timestamps_bf[id].strftime("%Y-%m-%d")}, '
            logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} ({print_str})')
        else:
            # fill in all zero
            if interpolation == 'zero' or i == 1:
                band_list = np.zeros((meta['height'] * meta['width'], meta['count']))
                logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} (0)')
            # fill in the last band_list
            else:
                band_list = bands_list[-1]
                logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} (previous)')
        bands_list.append(band_list)
    # stack finally
    bands_array = np.stack(bands_list, axis=2)
    logger.info('Satck done!')

    return bands_array, meta, timestamps_bf, timestamps_af, timestamps_ref


def count_classes(logger, y):
    tot_num = len(y)
    for i in np.unique(y):
        y_i = y[y == i]
        logger.info(f'  label = {i}, pixel number = {y_i.shape[0]}, percentage = {round(len(y_i)/tot_num*100, 2)}%')


def save_predictions_geotiff(meta_src, predictions, save_path):
    # Register GDAL format drivers and configuration options with a context manager
    with rasterio.Env():
        # Write an array as a raster band to a new 8-bit file. We start with the profile of the source
        out_meta = meta_src.copy()
        out_meta.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw')
        with rasterio.open(save_path, 'w', **out_meta) as dst:
            # reshape into (band, height, width)
            dst.write(predictions.reshape(1, out_meta['height'], out_meta['width']).astype(rasterio.uint8))


def impurity_importance_table(feature_names, feature_importance, save_path):
    df = pd.DataFrame()
    df['feature_names'] = feature_names
    df['feature_importance'] = feature_importance
    df.sort_values(by=['feature_importance']).to_csv(save_path, index=False)


def permutation_importance_table(model, x_val, y_val, feature_names, save_path):
    r = permutation_importance(model, x_val, y_val, n_repeats=30, random_state=0)
    df = pd.DataFrame()
    for i in r.importances_mean.argsort()[::-1]:
        df['feature_names'] = feature_names[i]
        df['feature_importance'] = r.importances_mean[i]
        df['feature_importance_std'] = r.importances_std[i]
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{feature_names[i]:<8}" 
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")
    df.to_csv(save_path, index=False)


def dropna_in_shapefile(from_shp_path, to_shp_path=None):
    shapefile = gpd.read_file(from_shp_path)
    shapefile = shapefile.dropna().reset_index(drop=True)
    if to_shp_path is None:
        shapefile.to_file(from_shp_path)
    else:
        shapefile.to_file(to_shp_path)


"""
=============================== labels ================================
"""


def load_target_shp(path, transform=None, proj_out=None):
    """ Load the shapefile as a list of numpy array of coordinates
        INPUT : path (str) -> the path to the shapefile
                transform (rasterio.Affine) -> the affine transformation to get the polygon in row;col format from UTM.
        OUTPUT : poly (list of np.array) -> list of polygons (as numpy.array of coordinates)
                 poly_rc (list of np.array) -> list of polygon in row-col format if a transform is given
    """
    with fiona.open(path) as shapefile:
        proj_in = pyproj.Proj(shapefile.crs)
        class_type = [feature['properties']['id'] for feature in shapefile]
        features = [feature["geometry"] for feature in shapefile]
    # re-project polygons if necessary
    if proj_out is None or proj_in == proj_out:
        poly = [np.array([(coord[0], coord[1]) for coord in features[i]['coordinates'][0]]) for i in
                range(len(features))]
        print('No re-projection!')
    else:
        poly = [np.array(
            [pyproj.transform(proj_in, proj_out, coord[0], coord[1]) for coord in features[i]['coordinates'][0]]) for i
                in range(len(features))]
        print(f'Re-project from {proj_in} to {proj_out}')

    poly_rc = None
    # transform in row-col if a transform is given
    if transform is not None:
        poly_rc = [np.array([rasterio.transform.rowcol(transform, coord[0], coord[1])[::-1] for coord in p]) for p in
                   poly]
    print('Loaded target shape files.')

    return poly, poly_rc, class_type


def compute_mask(polygon_list, meta, val_list):
    """ Get mask of class of a polygon list
        INPUT : polygon_list (list od polygon in coordinates (x, y)) -> the polygons in row;col format
                meta -> the image width and height
                val_list(list of int) -> the class associated with each polygon
        OUTPUT : img (np.array 2D) -> the mask in which the pixel value reflect it's class (zero being the absence of class)
    """
    img = np.zeros((meta['height'], meta['width']), dtype=np.uint8)  # skimage : row,col --> h,w
    for polygon, val in zip(polygon_list, val_list):
        rr, cc = skimage.draw.polygon(polygon[:, 1], polygon[:, 0], img.shape)
        img[rr, cc] = val
    print("Added targets' mask.")
    return img


"""
=============================== logger ================================
"""


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def get_log_dir(log_dir='../logs/'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
