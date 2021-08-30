import argparse
import os

import numpy as np

import skimage
import geopandas as gpd
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling


def calculate_ndvi(red, nir):
    """ Compute the NDVI
        INPUT : red (np.array) -> the Red band images as a numpy array of float
                nir (np.array) -> the Near Infrared images as a numpy array of float
        OUTPUT : ndvi (np.array) -> the NDVI
    """
    ndvi = (nir - red) / (nir + red + 1e-12)
    return ndvi


def calculate_ndre(red_edge, nir):
    ndre = (nir - red_edge) / (nir + red_edge + 1e-12)
    return ndre


def calculate_gndvi(green, nir):
    gndvi = (nir - green) / (nir + green + 1e-12)
    return gndvi


def calculate_evi(green, red, blue, nir):
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    return evi


def calculate_cvi(green, red, nir):
    cvi = nir * red / (green + 1e-12)**2
    return cvi


def add_features(img, new_features=['ndvi', 'gndvi']):
    """
    band02 = blue --> idx = 0
    band03 = green --> idx = 1
    band04 = red --> idx = 2
    band08 = nir --> idx = 3
    """
    new_bands = []

    # select specific bands
    # img = img[:, :, [1, 2, 3, 4]]

    # bands
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    nir = img[:, :, 3]

    # add feature
    for feature in new_features:
        if feature == 'ndvi':
            new_bands.append(calculate_ndvi(red, nir))
        elif feature == 'gndvi':
            new_bands.append(calculate_gndvi(green, nir))
        elif feature == 'evi':
            new_bands.append(calculate_evi(green, blue, blue, nir))
        elif feature == 'cvi':
            new_bands.append(calculate_cvi(green, red, nir))

    return np.append(img, np.stack(new_bands, axis=2), axis=2)


def load_geotiff(path, window=None):
    """ Load the geotiff as a list of numpy array.
        INPUT : path (str) -> the path to the geotiff
                window (raterio.windows.Window) -> the window to use when loading the image
        OUTPUT : band (list of numpy array) -> the different bands as float scalled to 0:1
                 meta (dictionnary) -> the metadata associated with the geotiff
    """
    with rasterio.open(path) as f:
        band = [skimage.img_as_float(f.read(i+1, window=window)) for i in range(f.count)]
        meta = f.meta
        if window != None:
            meta['height'] = window.height
            meta['width'] = window.width
            meta['transform'] = f.window_transform(window)
    return band, meta


def upsampling_20m_to_10m():
    pass


def read_shp(shp_name):
    return gpd.read_file(shp_name)


def prepare_labels(label_path):
    pass


def split_raster():
    pass


def clip_all_rasters(images_dir):
    # shape file information
    shape_filepath = '../data/study-area/study_area.shp'
    with fiona.open(shape_filepath, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    shape_crs = gpd.read_file(shape_filepath).crs

    # geotiff directory
    geotiff_dir = images_dir + 'geotiff/'
    if not os.path.exists(geotiff_dir):
        os.mkdir(geotiff_dir)
    # clip directory
    clip_dir = images_dir + 'clip/'
    if not os.path.exists(clip_dir):
        os.mkdir(clip_dir)

    # clip all the raster
    for filename in [f for f in os.listdir(geotiff_dir) if f.endswith('.tiff')]:
        geotiff_filepath = geotiff_dir + filename
        clip_filepath = clip_dir + filename
        clip_single_raster(shape_crs, shapes, geotiff_filepath, clip_filepath)


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

    with rasterio.open(clip_filepath, "w", **out_meta) as dest:
        dest.write(out_image)


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


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # configuration
    parser.add_argument('--images_dir', type=str,
                        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/')
    args = parser.parse_args()
    main(args)