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


# def stack_all_timestamps(from_dir):
#     # stack images of different timestamp
#     stacked_filenames = []
#     stacked_raster = dict(band=[], meta=[], timestamp=[])
#     for filename in os.listdir(from_dir):
#         # read file
#         raster_filepath = from_dir + filename
#         band, meta = load_geotiff(raster_filepath)
#         # check if all black. yes -> discard, no -> continue
#         if np.array(band).mean() != 0.0:
#             stacked_filenames.append(raster_filepath)
#             # get the data
#             timestamp = re.split('[_.]', filename)[-2]
#             stacked_raster['band'].append(np.stack(band, axis=2))
#             stacked_raster['meta'] = meta
#             stacked_raster['timestamp'].append(datetime.strptime(timestamp, '%Y%m%dT%H%M%S%f'))
#     return stacked_raster


def timestamp_sanity_check(timestamp_std, filename):
    timestamp_get = datetime.strptime(re.split('[_.]', filename)[-2],
                                      '%Y%m%dT%H%M%S%f').date()
    timestamp_get = timestamp_get - timedelta(days=timestamp_get.weekday())
    if timestamp_std != timestamp_get:
        print(f'{timestamp_std} and {timestamp_get} do not match!')


def get_weekly_timestamps():
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


def equidistant_stack(from_dir):

    print('*** Stacking equidistantly ***')

    # sorted available files
    filenames = sorted(os.listdir(from_dir))

    # find all the time stamps in "from" folder
    timestamps_bf = []
    for filename in filenames:
        timestamps_bf.append(datetime.strptime(re.split('[_.]', filename)[-2],
                                               '%Y%m%dT%H%M%S%f').date())
    # find all corresponding Monday, thus equally spaced
    timestamps_af = [ts - timedelta(days=ts.weekday()) for ts in timestamps_bf]
    # date of all Monday in 2020
    timestamps_weekly = get_weekly_timestamps()

    # get bands' meta data
    _, meta = load_geotiff(from_dir + os.listdir(from_dir)[0])

    # stack equidistantly
    timestamps_af_pd = pd.Series(timestamps_af)
    bands_list = []
    for i, timestamp in enumerate(timestamps_weekly, start=1):
        # get all the indices of that week
        ids = timestamps_af_pd[timestamps_af_pd.eq(timestamp)].index
        # with empty week
        if len(ids) == 0:
            band_list = []
            # band_list = np.zeros((meta['height'] * meta['width'], meta['count']))
            # print(f'[{i}/{len(timestamps_weekly)}] {timestamp} (none)')
        # with single or multiple week
        else:
            band_list, black_ids = [], []
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
                    print(f' Discard {filename} due to 0 values.')
            if len(band_list) != 0:
                band_list = np.stack(band_list, axis=2).mean(axis=2)
                # format printing
                print(f'[{i}/{len(timestamps_weekly)}] {timestamp} (', end=" ")
                for id in ids:
                    if id in black_ids:
                        print(f'x{timestamps_bf[id]}', end=", ")
                    else:
                        print(timestamps_bf[id], end=", ")
                print(')', end='\n')
            else:
                # band_list = np.zeros((meta['height'] * meta['width'], meta['count']))  # fill in 0
                band_list = bands_list[-1]  # fill in the last band_list
                print(f'[{i}/{len(timestamps_weekly)}] {timestamp} (last bands)')
        bands_list.append(band_list)
    bands_array = np.stack(bands_list, axis=2)

    print('\tDone!')

    return bands_array, meta, timestamps_bf, timestamps_af, timestamps_weekly


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
