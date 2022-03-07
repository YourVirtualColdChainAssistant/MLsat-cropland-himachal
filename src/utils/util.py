import os
import fiona
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling


def load_shp_to_array(shp_path, meta):
    print("Loading shapefile to nd.array...")
    shp_gdf = gpd.read_file(shp_path).to_crs(meta['crs'])
    val_list = list(shp_gdf.id.values)
    features_list = [shapely.geometry.mapping(s) for s in shp_gdf.geometry]

    iterable = iter([(feature, val) for feature, val in zip(features_list, val_list)])
    img = rasterio.features.rasterize(iterable, out_shape=(meta['height'], meta['width']),
                                      transform=meta['transform'])
    return features_list, img


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


def find_file(string, search_path):
    result = []
    # Walking top-down from the root
    for root, _, files in os.walk(search_path):
        for filename in files:
            if string in filename:
                result.append(os.path.join(root, filename))
    return result


def find_folder(string, search_path):
    result = []
    # Walking top-down from the root
    for root, dirs, _ in os.walk(search_path):
        for d in dirs:
            if string in d:
                result.append(os.path.join(root, d))
    return result


def find_top_level(string, search_path):
    result = []
    for filename in os.listdir(search_path):
        if string in filename:
            result.append(os.path.join(search_path, filename))
    return result


def save_cv_results(cv_results, save_path):
    df = pd.DataFrame()
    # save result
    for i in ['params', 'mean_test_accuracy', 'std_test_accuracy', 'rank_test_accuracy',
              'mean_test_precision', 'std_test_precision', 'rank_test_precision',
              'mean_test_recall', 'std_test_recall', 'rank_test_recall',
              'mean_test_f1_score', 'std_test_f1_score', 'rank_test_f1_score',
              'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']:
        df[i] = cv_results[i]
    df.to_csv(save_path, index=False, header=True)


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
