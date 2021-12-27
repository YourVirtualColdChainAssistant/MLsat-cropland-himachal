import os
import fiona
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import skimage
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling


def load_geotiff(path, window=None, read_as='as_integer'):
    """ Load the geotiff as a list of numpy array.
        INPUT : path (str) -> the path to the geotiff
                window (rasterio.windows.Window) -> the window to use when loading the image
        OUTPUT : band (list of numpy array) -> the different bands unscaled
                 meta (dictionary) -> the metadata associated with the geotiff
    """
    with rasterio.open(path) as f:
        if read_as == 'as_float':
            band = [skimage.img_as_float(f.read(i + 1, window=window)) for i in range(f.count)]
        elif read_as == 'as_reflectance':
            band = [f.read(i + 1, window=window) / 10000 for i in range(f.count)]
        else:  # normal read as integer
            band = [f.read(i + 1, window=window) for i in range(f.count)]
        meta = f.meta
        if window is not None:
            meta['height'] = window.height
            meta['width'] = window.width
            meta['transform'] = f.window_transform(window)
    return band, meta


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


def load_shp_to_array(shp_path, meta):
    print("Loading shapefile to nd.array...")
    shp_gdf = gpd.read_file(shp_path).to_crs(meta['crs'])
    val_list = list(shp_gdf.id.values)
    features_list = [shapely.geometry.mapping(s) for s in shp_gdf.geometry]

    iterable = iter([(feature, val) for feature, val in zip(features_list, val_list)])
    img = rasterio.features.rasterize(iterable, out_shape=(meta['height'], meta['width']),
                                      transform=meta['transform'])
    print('  ok')
    return features_list, val_list, img


def convert_gdf_to_shp(data, save_path):
    with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
        data.to_file(save_path)


def read_shp(file_path):
    with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
        data = gpd.read_file(file_path)
    return data


def count_classes(logger, y):
    tot_num = len(y)
    for i in np.unique(y):
        y_i = y[y == i]
        msg = f'  label = {i}, pixel number = {y_i.shape[0]}, percentage = {round(len(y_i) / tot_num * 100, 2)}%'
        if logger is None:
            print(msg)
        else:
            logger.info(msg)


def dropna_in_shapefile(from_shp_path, to_shp_path=None):
    shapefile = gpd.read_file(from_shp_path)
    shapefile = shapefile.dropna().reset_index(drop=True)
    if to_shp_path is None:
        shapefile.to_file(from_shp_path)
    else:
        shapefile.to_file(to_shp_path)


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
            for polygon in multi_polygon.geoms:
                new_polygons.append([label, district, polygon])
        new_polygons_df = pd.DataFrame(new_polygons, columns=['id', 'district', 'geometry'])
        polygons_df = pd.concat([polygons_df, new_polygons_df], axis=0)
    return polygons_df


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


def merge_tiles(path_list, out_path, bounds=None):
    """ Merge the raster images given in the path list and save the results on disk.
        INPUT : path_list (list of string) -> the path to all the images to merge
                out_path (str) -> the path and file name to which the merge is saved
                bounds (tuple) -> (left, bottom, right, top) the boundaries to extract from (in UTM).
        OUTPUT : None
    """
    # open all tiles
    src_file_mosaic = []
    for fpath in path_list:
        src = rasterio.open(fpath)
        src_file_mosaic.append(src)
    # merge the files into a single mosaic
    mosaic, out_trans = rasterio.merge.merge(src_file_mosaic, bounds=bounds)
    # update
    out_meta = src.meta.copy()
    out_meta.update(
        {'driver': 'GTiff', 'height': mosaic.shape[1], 'width': mosaic.shape[2], 'transform': out_trans})
    # save the merged
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(mosaic)


def prepare_meta_window_descriptions(geotiff_dir, label_path):
    img_path = geotiff_dir + os.listdir(geotiff_dir)[0]
    img = rasterio.open(img_path)
    transform, dst_crs, descriptions = img.transform, img.crs, img.descriptions

    label_shp = gpd.read_file(label_path)
    label_shp = label_shp.to_crs(dst_crs)
    window = get_window(transform, label_shp.total_bounds)

    _, meta = load_geotiff(img_path, window, read_as='as_raw')
    return meta, window, descriptions


def prepare_meta_descriptions(geotiff_dir, window):
    img_path = geotiff_dir + os.listdir(geotiff_dir)[0]
    img = rasterio.open(img_path)
    transform, dst_crs, descriptions = img.transform, img.crs, img.descriptions

    _, meta = load_geotiff(img_path, window, read_as='as_raw')
    return meta, descriptions


def get_window(transform, bounds):
    """

    Parameters
    ----------
    transform
    bounds

    Returns
    -------
    Window(col_off, row_off, width, height)
    """
    minx, miny, maxx, maxy = bounds
    row_min, col_min = rasterio.transform.rowcol(transform, minx, maxy)
    row_max, col_max = rasterio.transform.rowcol(transform, maxx, miny)
    return Window(col_min, row_min, col_max - col_min, row_max - row_min)


def resample(in_file, h_target, w_target):
    with rasterio.open(in_file) as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(dataset.count, h_target, w_target),
            resampling=Resampling.nearest
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / w_target),
            (dataset.height / h_target)
        )

        return data, transform


def find_file(string, search_path):
    result = []
    # Wlaking top-down from the root
    for root, dir_, files in os.walk(search_path):
        for f in files:
            if string in f:
                filename = f
                result.append(os.path.join(root, filename))
                break
    return result
