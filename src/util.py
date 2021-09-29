import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import geopandas as gpd
import fiona
import skimage
import pyproj
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from preprocessing import add_features


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
    timestamp_get = datetime.strptime(re.split('[_.]', filename)[-2],
                                      '%Y%m%dT%H%M%S%f').date()
    timestamp_get = timestamp_get - timedelta(days=timestamp_get.weekday())
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
    date_start = date(2020, 1, 1)
    date_end = date(2020, 12, 31)
    date_start = date_start - timedelta(days=date_start.weekday())
    date_end = date_end - timedelta(days=date_end.weekday())
    d = date_start
    weekly_timestamps = []
    while d <= date_end:
        weekly_timestamps.append(d)
        d += timedelta(7)
    return weekly_timestamps


def get_monthly_timestamps():
    """
    Get the first day of each month in 2020.

    """
    date_start = date(2020, 1, 1)
    monthly_timestamps = []
    for m in range(1, 13):
        monthly_timestamps.append(date_start.replace(month=m))
    return monthly_timestamps


def stack_valid_raw_ndvi(from_dir):
    print('Stacking valid NDVI...')
    bands_list_raw, timestamps_raw, timestamps_missing, meta = \
        stack_valid_raw_timestamps(from_dir)
    bands_array_raw = add_features(np.stack(bands_list_raw, axis=2), new_features=['ndvi'])
    ndvi_array_raw = bands_array_raw[:, -1, :]
    return ndvi_array_raw, timestamps_raw, meta


def stack_equidistant_ndvi(ndvi_array_raw, timestamps_raw, meta, way, interpolation='previous'):
    choices_sanity_check(['weekly', 'monthly'], way, 'way')
    if way == 'weekly':
        timestamps_af = [ts - timedelta(days=ts.weekday()) for ts in timestamps_raw]  # datetime.weekday() returns 0~6
        timestamps_ref = get_weekly_timestamps()
    else:
        timestamps_af = [ts - timedelta(days=ts.day-1) for ts in timestamps_raw]  # datetime.day returns 1-31
        timestamps_ref = get_monthly_timestamps()

    # stack equidistantly
    timestamps_af_pd = pd.Series(timestamps_af)
    ndvi_list_eql = []
    for i, timestamp in enumerate(timestamps_ref, start=1):
        # get all the indices
        ids = list(timestamps_af_pd[timestamps_af_pd.eq(timestamp)].index)
        # with non-empty data
        if len(ids) != 0:
            ndvi_array = ndvi_array_raw[:, ids].max(axis=1)
            # format printing
            print(f'[{i}/{len(timestamps_ref)}] {timestamp} (', end=" ")
            for id in ids:
                print(timestamps_raw[id], end=", ")
            print(')', end='\n')
        else:
            if interpolation == 'zero' or i == 1:
                ndvi_array = np.zeros(meta['height'] * meta['width'])
                print(f'[{i}/{len(timestamps_ref)}] {timestamp} (0)')
            else:
                ndvi_array = ndvi_list_eql[-1]
                print(f'[{i}/{len(timestamps_ref)}] {timestamp} (previous)')
        ndvi_list_eql.append(ndvi_array)
    ndvi_array_eql = np.stack(ndvi_list_eql, axis=1)

    return ndvi_array_eql, timestamps_af, timestamps_ref


def stack_valid_raw_timestamps(from_dir):
    print('----- Stacking valid raw timestamps -----')
    bands_list_valid, timestamps_valid, timestamps_missing = [], [], []
    for filename in sorted(os.listdir(from_dir)):
        raster_filepath = from_dir + filename
        band, meta = load_geotiff(raster_filepath)
        # pixel values check
        if np.array(band).mean() != 0.0:
            bands_list_valid.append(np.stack(band, axis=2).reshape(-1, len(band)))
            timestamps_valid.append(datetime.strptime(re.split('[_.]', filename)[-2],
                                                    '%Y%m%dT%H%M%S%f').date())
        else:
            timestamps_missing.append(datetime.strptime(re.split('[_.]', filename)[-2],
                                                    '%Y%m%dT%H%M%S%f').date())
            print(f'\tDiscard {filename} due to empty value.')
    print('Done!')
    return bands_list_valid, timestamps_valid, timestamps_missing, meta


def stack_all_timestamps(from_dir, way='weekly', interpolation='previous'):
    """
    Stack all the timestamps in from_dir folder, ignoring all black images.
    :param from_dir: string
    :param way: string
        choices = ['raw', 'weekly', 'monthly']
    :param interpolation: string
        choices = ['zero', 'previous']
    :return: bands_array, meta, timestamps_bf, timestamps_af, timestamps_weekly
        timestamps_bf: the raw timestamps
    """

    print(f'----- Stacking all timestamps {way} -----')

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
        timestamps_bf.append(datetime.strptime(re.split('[_.]', filename)[-2],
                                               '%Y%m%dT%H%M%S%f').date())

    # ### check the way to stack
    if way == 'raw':
        timestamps_af = timestamps_bf
        timestamps_ref = timestamps_bf
    elif way == 'weekly':
        timestamps_af = [ts - timedelta(days=ts.weekday()) for ts in
                         timestamps_bf]  # datetime.weekday() returns 0~6
        timestamps_ref = get_weekly_timestamps()
    else:
        timestamps_af = [ts - timedelta(days=ts.day-1) for ts in timestamps_bf]  # datetime.day returns 1-31
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
                    print(f'\tDiscard {filename} due to empty value.')
        # stack by index
        if len(band_list) != 0:
            band_list = np.stack(band_list, axis=2).mean(axis=2)
            # format printing
            print(f'[{i}/{len(timestamps_ref)}] {timestamp} (', end=" ")
            for id in ids:
                if id in black_ids:
                    print(f'x{timestamps_bf[id]}', end=", ")
                else:
                    print(timestamps_bf[id], end=", ")
            print(')', end='\n')
        else:
            # fill in all zero
            if interpolation == 'zero' or i == 1:
                band_list = np.zeros((meta['height'] * meta['width'], meta['count']))
                print(f'[{i}/{len(timestamps_ref)}] {timestamp} (0)')
            # fill in the last band_list
            else:
                band_list = bands_list[-1]
                print(f'[{i}/{len(timestamps_ref)}] {timestamp} (last bands)')
        bands_list.append(band_list)
    # stack finally
    bands_array = np.stack(bands_list, axis=2)
    print('Stack is done!')

    return bands_array, meta, timestamps_bf, timestamps_af, timestamps_ref


def count_classes(y):
    for i in np.unique(y):
        print(f'Number of pixel taking {i} is {y[y==i].shape[0]}')


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


def feature_importance_table(feature_name, feature_importance, save_path):
    df = pd.DataFrame()
    df['feature_name'] = feature_name
    df['feature_importance'] = feature_importance
    df.sort_values(by=['feature_importance']).to_csv(save_path, index=False)
