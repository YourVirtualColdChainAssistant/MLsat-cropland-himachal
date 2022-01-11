import os
import skimage
import rasterio
import pandas as pd
import geopandas as gpd

from src.utils.util import multipolygons_to_polygons


def load_geotiff(path, window=None, read_as='as_integer'):
    """
    Load the geotiff as a list of numpy array.

    Parameters
    ----------
    path: string
        The path to the geotiff.
    window: rasterio.windows.Window
        The window to use when loading the image.
    read_as: choices = [as_integer, as_float, as_reflectance]

    Returns
    -------
    band: list of np.array
        The different bands unscaled.
    meta: dict
        The metadata associated with the geotiff.
    """
    with rasterio.open(path) as f:
        if read_as == 'as_float':
            band = [skimage.img_as_float(f.read(i + 1, window=window)) for i in range(f.count - 1)]
        elif read_as == 'as_reflectance':
            band = [f.read(i + 1, window=window) / 10000 for i in range(f.count - 1)]
        else:  # normal read as integer
            band = [f.read(i + 1, window=window) for i in range(f.count - 1)]
        band.append(f.read(f.count, window=window))
        meta = f.meta
        if window is not None:
            meta['height'] = window.height
            meta['width'] = window.width
            meta['transform'] = f.window_transform(window)
    return band, meta


def clean_train_shapefiles(save_to_path='./data/ground_truth/train_labels/train_labels.shp'):
    # read all the shape files
    print(os.path.abspath(os.getcwd()))
    old_apples_shp = gpd.read_file('./data/ground_truth/apples/survey20210716_polygons20210819_corrected20210831.shp')
    new_apples_shp = gpd.read_file('./data/ground_truth/apples/survey20210825_polygons20210901_revised20210929.shp')
    non_crops_shp = gpd.read_file('./data/ground_truth/non_crops/non_crops.shp')
    other_crops_shp = gpd.read_file('./data/ground_truth/other_crops/other_crops.shp')
    train_region_shp = gpd.read_file('./data/ground_truth/train_region/train_region.shp')
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


def clean_test_near_shapefiles(save_to_path='./data/ground_truth/test_labels_kullu/test_labels_kullu.shp'):
    label_shp = gpd.read_file('./data/ground_truth/test_polygons_near/test_polygons_near.shp')
    test_region_shp = gpd.read_file('./data/ground_truth/test_region_near/test_region_near.shp')
    # delete empty polygons and split multipolygons
    label_shp = label_shp.dropna().reset_index(drop=True)
    label_shp = multipolygons_to_polygons(label_shp)
    # mask for the study area
    save_label_in_region(label_shp, test_region_shp, save_to_path)


def clean_test_far_shapefiles():
    old_apples_shp = gpd.read_file('./data/ground_truth/apples/survey20210716_polygons20210819_corrected20210831.shp')
    new_apples_shp = gpd.read_file('./data/ground_truth/apples/survey20210825_polygons20210901_revised20210929.shp')
    # include other crops and non-cropland
    far_shp = gpd.read_file('./data/ground_truth/test_polygons_far/test_polygons_far.shp')
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
    mandi_path = './data/ground_truth/test_labels_mandi/test_labels_mandi.shp'
    shimla_path = './data/ground_truth/test_labels_shimla/test_labels_shimla.shp'
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
