import os
import re
import sys
import fiona
import pickle
import logging
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry
import skimage
import skimage.draw
import pyproj
import rasterio
from rasterio.mask import mask


def load_geotiff(path, window=None):
    """ Load the geotiff as a list of numpy array.
        INPUT : path (str) -> the path to the geotiff
                window (rasterio.windows.Window) -> the window to use when loading the image
        OUTPUT : band (list of numpy array) -> the different bands as float scaled to 0:1
                 meta (dictionary) -> the metadata associated with the geotiff
    """
    with rasterio.open(path) as f:
        band = [skimage.img_as_float(f.read(i + 1, window=window)) for i in range(f.count)]
        meta = f.meta
        if window is not None:
            meta['height'] = window.height
            meta['width'] = window.width
            meta['transform'] = f.window_transform(window)
    return band, meta


def clip_raster(img_dir, clip_from_shp):
    # check directory
    geotiff_dir = img_dir + 'geotiff/'
    clip_dir = img_dir + clip_from_shp.split('/')[2] + '/'
    if not os.path.exists(clip_dir):
        os.mkdir(clip_dir)

    # get all files
    clip_names = [f for f in sorted(os.listdir(geotiff_dir)) if f.endswith('.tiff')]

    # get the target crs (raster's crs rather than shp's crs)
    with rasterio.open(geotiff_dir + clip_names[0]) as src0:
        raster_crs = src0.crs
    # get the shp crs
    shp = gpd.read_file(clip_from_shp)
    # reproject shp if needed
    if shp.crs != raster_crs:
        shp = reproject_shapefile(raster_crs, shp)
    # get geojson-like shapes
    shapes = [shapely.geometry.mapping(s) for s in shp.geometry if s is not None]

    print('\nStart clipping...')
    for i, clip_name in enumerate(clip_names, start=1):
        geotiff_path = geotiff_dir + clip_name
        clip_path = clip_dir + clip_name
        clip_flag = clip_single_raster(shapes, geotiff_path, clip_path)
        if clip_flag:
            print(f'[{i}/{len(clip_names)}] Clipped {clip_path}')
        else:
            print(f'[{i}/{len(clip_names)}] Discarded {clip_path}')
    print(f'Clip done!')


def clip_single_raster(shapes, geotiff_path, clip_path):
    """

    :param shapes: geometry of source shapefile
    :param geotiff_path: xx.tiff path
    :param clip_path: output clipped path
    :return:
    """
    # read imagery file
    with rasterio.open(geotiff_path) as src:
        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta

    # Save clipped imagery
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    clip_flag = False
    if out_image.mean() != 0.0:
        with rasterio.open(clip_path, "w", **out_meta) as dst:
            dst.write(out_image)
        clip_flag = True
    return clip_flag


def reproject_shapefile(dst_crs, shp):
    return shp.to_crs(dst_crs)


def convert_gdf_to_shp(data, save_path):
    with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
        data.to_file(save_path)


def read_shp(file_path):
    with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
        data = gpd.read_file(file_path)
    return data


def timestamp_sanity_check(timestamp_std, filename):
    timestamp_get = datetime.datetime.strptime(re.split('[_.]', filename)[-2],
                                               '%Y%m%dT%H%M%S%f').date()
    timestamp_get = timestamp_get - datetime.timedelta(days=timestamp_get.weekday())
    if timestamp_std != timestamp_get:
        raise ValueError(f'{timestamp_std} and {timestamp_get} do not match!')


def choices_sanity_check(choices, choice, var_name):
    if choice not in choices:
        raise ValueError(f'{choice} is unavailable. Please choose "{var_name}" from {choices}')


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
        shape (n_pixels, n_bands, n_weeks)
    timestamps_bf: the raw timestamps
    """

    # ### sanity check
    choices_sanity_check(['raw', 'weekly', 'monthly'], way, 'way')
    choices_sanity_check(['zero', 'previous'], interpolation, 'interpolation')

    # ### raw files
    # sorted available files
    filenames = sorted([file for file in os.listdir(from_dir) if file.endswith('tiff')])
    # get bands' meta data
    _, meta = load_geotiff(from_dir + filenames[0])
    # find all the raw time stamps
    timestamps_bf = []
    for filename in filenames:
        timestamps_bf.append(datetime.datetime.strptime(re.split('[_.]', filename)[-2], '%Y%m%dT%H%M%S%f').date())

    # ### check the way to stack
    if way == 'raw':
        timestamps_af = timestamps_bf
        timestamps_ref = timestamps_bf
    elif way == 'weekly':
        timestamps_af = [ts - datetime.timedelta(days=ts.weekday()) for ts in
                         timestamps_bf]  # datetime.weekday() returns 0~6
        timestamps_ref = get_weekly_timestamps()
    else:
        timestamps_af = [ts - datetime.timedelta(days=ts.day - 1) for ts in timestamps_bf]  # datetime.day returns 1-31
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

        # stack by index
        if len(band_list) != 0:
            band_list = np.stack(band_list, axis=2).max(axis=2)  # take max
        elif interpolation == 'zero' or i == 1:
            band_list = np.zeros((meta['height'] * meta['width'], meta['count']))
        else:
            band_list = bands_list[-1]

        # print
        if len(ids) != 0:
            print_str = ''
            for id in ids:
                if id in black_ids:
                    print_str += f'x{timestamps_bf[id].strftime("%Y-%m-%d")}, '
                else:
                    print_str += f'{timestamps_bf[id].strftime("%Y-%m-%d")}, '
            logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} ({print_str})')
        elif interpolation == 'zero' or i == 1:
            logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} (0)')
        else:
            logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} (previous)')

        bands_list.append(band_list)
    # stack finally
    bands_array = np.stack(bands_list, axis=2)
    logger.info('  ok')

    return bands_array, meta, timestamps_bf, timestamps_af, timestamps_ref


def count_classes(logger, y):
    tot_num = len(y)
    for i in np.unique(y):
        y_i = y[y == i]
        msg = f'  label = {i}, pixel number = {y_i.shape[0]}, percentage = {round(len(y_i) / tot_num * 100, 2)}%'
        if logger is None:
            print(msg)
        else:
            logger.info(msg)


def save_predictions_geotiff(meta_src, predictions, save_path):
    # Register GDAL format drivers and configuration options with a context manager
    color_map = {
        0: (0, 0, 0),
        1: (240, 65, 53),
        2: (154, 205, 50),
        3: (184, 134, 11),
        255: (255, 255, 255)
    }
    with rasterio.Env():
        # Write an array as a raster band to a new 8-bit file. We start with the profile of the source
        out_meta = meta_src.copy()
        out_meta.update(
            dtype=rasterio.uint8,
            count=1)
        with rasterio.open(save_path, 'w', **out_meta) as dst:
            # reshape into (band, height, width)
            dst.write(predictions.reshape(1, out_meta['height'], out_meta['width']).astype(rasterio.uint8))
            dst.write_colormap(1, color_map)


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
    print("Loading target shapefile...")
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

    return features, poly, poly_rc, class_type


def compute_mask(polygon_list, meta, val_list):
    """ Get mask of class of a polygon list
        INPUT : polygon_list (list od polygon in coordinates (x, y)) -> the polygons in row;col format
                meta -> the image width and height
                val_list(list of int) -> the class associated with each polygon
        OUTPUT : img (np.array 2D) -> the mask in which the pixel value reflect it's class (zero being the absence of class)
    """
    img = np.zeros((meta['height'], meta['width']), dtype=np.uint8)  # skimage : row,col --> h,w
    i = 0
    for polygon, val in zip(polygon_list, val_list):
        rr, cc = skimage.draw.polygon(polygon[:, 1], polygon[:, 0], img.shape)
        img[rr, cc] = val
        i += 1
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


def multipolygons_to_polygons(shp_file):
    # check the number of multipolygons
    multi_polygons_df = shp_file[shp_file['geometry'].type == 'MultiPolygon']
    polygons_df = shp_file[shp_file['geometry'].type == 'Polygon']
    if multi_polygons_df.shape[0] == 0:
        print('No multi-polygons!')
    else:
        new_polygons = []
        num_multi_polygons = multi_polygons_df.shape[0]
        print(f'Converting {num_multi_polygons} multi-polygons to polygons...')
        for i in range(num_multi_polygons):
            multi_polygon_ = multi_polygons_df.iloc[i]
            label, district, multi_polygon = multi_polygon_.id, multi_polygon_.district, multi_polygon_.geometry
            for polygon in list(multi_polygon):
                new_polygons.append([label, district, polygon])
        new_polygons_df = pd.DataFrame(new_polygons, columns=['id', 'district', 'geometry'])
        polygons_df = pd.concat([polygons_df, new_polygons_df], axis=0)
    return polygons_df


def correct_predictions():
    pass


def resample_negatives(pos, neg):
    pass


def get_cropland_mask(df, n_feature, pretrained_name):
    # read pretrained model
    trained_model = pickle.load(open(f'../models/{pretrained_name}.sav', 'rb'))

    # new columns
    df['gt_cropland'] = 0
    df.loc[(df.label.values == 1) | (df.label.values == 2), 'gt_cropland'] = 1

    # cropland
    to_pred_mask = df.label.values == 0
    preds = trained_model.predict(df.iloc[to_pred_mask, :n_feature].values)
    cropland_pred = np.empty_like(preds)
    cropland_pred[preds == 2] = 1
    cropland_pred[preds == 3] = 0
    df['gp_cropland'] = df['gt_cropland'].copy()
    df.loc[to_pred_mask, 'gp_cropland'] = cropland_pred
    df['gp_cropland_mask'] = False
    df.loc[df.gp_cropland.values == 1, 'gp_cropland_mask'] = True

    return df.gp_cropland_mask.values


def save_cv_results(res, save_path):
    df = pd.DataFrame()
    df['mean_fit_time'] = res['mean_fit_time']
    df['std_fit_time'] = res['std_fit_time']
    df['mean_score_time'] = res['mean_score_time']
    df['std_score_time'] = res['std_score_time']
    df['params'] = res['params']
    df['mean_test_score'] = res['mean_test_score']
    df['std_test_score'] = res['std_test_score']
    df['rank_test_score'] = res['rank_test_score']
    df.to_csv(save_path, index=False)


def mosaic():
    pass
