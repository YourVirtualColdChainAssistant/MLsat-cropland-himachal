import os
import fiona
import shapely
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from sklearn.inspection import permutation_importance
from src.utils.util import load_geotiff
from src.utils.clip import clip_single_raster


def prepare_open_datasets(dataset_path, out_path, pred_path, region_shp_path, label_only=True):
    # clip to a bounding box, to reduce computation cost
    region_shp = gpd.read_file(region_shp_path)
    minx, miny, maxx, maxy = region_shp.total_bounds
    box_shp = gpd.GeoDataFrame({'geometry': shapely.geometry.box(minx, miny, maxx, maxy)}, index=[0])
    box_shp = box_shp.set_crs(region_shp.crs)
    inter_path = out_path.replace(out_path.split('_')[-1], 'intermediate_result.tiff')
    clip_open_datasets_to_shp(dataset_path, inter_path, box_shp)
    # align with same resolution
    align_raster(pred_path, inter_path, out_path)
    if label_only:
        clip_open_datasets_to_shp(out_path, out_path, region_shp)
    os.remove(inter_path)


def clip_open_datasets_to_shp(in_path, out_path, region_shp):
    with rasterio.open(in_path, 'r') as src0:
        in_crs = src0.crs
    if region_shp.crs != in_crs:
        region_shp = region_shp.to_crs(in_crs)
    region_shapes = [shapely.geometry.mapping(s) for s in region_shp.geometry if s is not None]
    clip_single_raster(region_shapes, in_path, out_path)


def align_raster(pred_path, in_path, out_path):
    """
    Align according to prediction file (with boundary and resolution adjustment).

    """
    # prepare source info
    bounds = rasterio.open(pred_path).bounds
    _, meta_tar = load_geotiff(pred_path)

    # command
    cmd = f"gdalwarp -overwrite -r average -t_srs {meta_tar['crs']} -ts {meta_tar['width']} {meta_tar['height']} " + \
          f"-te {bounds.left} {bounds.bottom} {bounds.right} {bounds.top} {in_path} {out_path}"
    returned_val = os.system(cmd)
    if returned_val == 0:
        print('Aligned raster!')
    else:
        raise ValueError('Alignment failed!')


def evaluate_by_gfsad(pred_path, dataset_path, logger=None):
    """
    Compare with GFSAD dataset to evaluate the overlap with croplands.
    Legend of GFSAD: 2 = croplands, 1 = non-croplands.
    Legend of Predictions: 2 = croplands, 3 = non-croplands.

    Parameters
    ----------
    pred_path: string
        path of predictions to compare
    dataset_path: string
        path of gfsad dataset
    logger:

    Returns
    -------

    """
    # load data
    band_pred, meta_pred = load_geotiff(pred_path, read_as='as_integer')
    band_dataset, meta_dataset = load_geotiff(dataset_path, read_as='as_integer')
    band_pred = band_pred[0]
    band_dataset = band_dataset[0]

    n_dataset = (band_dataset == 2).sum()
    n_pred = (band_pred[band_dataset == 2] == 2).sum()

    if logger is None:
        print(f'Cropland pixel number in GFASD: {n_dataset}')
        print(f'Cropland pixel number in prediction: {n_pred}')
        print(f'Percentage: {n_pred / n_dataset * 100:.2f}%')
    else:
        logger.info(f'Cropland pixel number in GFASD: {n_dataset}')
        logger.info(f'Cropland pixel number in prediction: {n_pred}')
        logger.info(f'Percentage: {n_pred / n_dataset * 100:.2f}%')


def evaluate_by_copernicus(pred_path, dataset_path, logger=None):
    """
    Compare with Copernicus dataset to evaluate the overlap of non-croplands.
    Legend of Copernicus: 50 = built-up, 111 = closed forest / evergreen needle leaf.
    Legend of Predictions: 2 = croplands, 3 = non-croplands.

    Parameters
    ----------
    pred_path: string
        path of predictions to compare
    dataset_path: string
        path of gfsad dataset
    logger

    Returns
    -------

    """
    # load data
    band_pred, meta_pred = load_geotiff(pred_path, read_as='as_integer')
    band_dataset, meta_dataset = load_geotiff(dataset_path, read_as='as_integer')
    band_pred = band_pred[0]
    band_dataset = band_dataset[0]

    # calculate
    n_dataset = ((band_dataset == 50) | (band_dataset == 111)).sum()
    n_pred = (band_pred[(band_dataset == 50) | (band_dataset == 111)] == 3).sum()
    if logger is None:
        print(f'Non-cropland pixel number in Copernicus: {n_dataset}')
        print(f'Non-cropland pixel number in prediction: {n_pred}')
        print(f'Percentage: {n_pred / n_dataset * 100:.2f}%')
    else:
        logger.info(f'Non-cropland pixel number in Copernicus: {n_dataset}')
        logger.info(f'Non-cropland pixel number in prediction: {n_pred}')
        logger.info(f'Percentage: {n_pred / n_dataset * 100:.2f}%')


def diff_two_predictions(pred_path_1, pred_path_2):
    # read
    band_pred_1, meta_pred_1 = load_geotiff(pred_path_1)
    band_pred_2, meta_pred_2 = load_geotiff(pred_path_2)
    # differentiate
    # TODO: different color for case [1,0] and [0,1]
    band_diff = np.zeros_like(band_pred_1[0])
    band_diff[band_pred_1[0] != band_pred_2[0]] = 1
    band_diff = np.expand_dims(band_diff, axis=0)
    # save name
    pred_name_1 = pred_path_1.split('/')[-1].split('.')[0]
    pred_name_2 = pred_path_2.split('/')[-1].split('.')[0]
    output_path = f"./preds/diff_{pred_name_1}_{pred_name_2}.tiff"
    # save
    with rasterio.open(output_path, "w", **meta_pred_1) as dst:
        dst.write(band_diff)
        print(f'Save difference between {pred_name_1} and {pred_name_2} to {output_path}')


def impurity_importance_table(feature_names, feature_importance, save_path):
    df = pd.DataFrame()
    df['feature_names'] = feature_names
    df['feature_importance'] = feature_importance
    df.sort_values(by=['feature_importance']).to_csv(save_path, index=False)


def permutation_importance_table(model, x_val, y_val, feature_names, save_path):
    print('permutation_importance... ')
    r = permutation_importance(model, x_val, y_val, random_state=0)
    print('permutation_importance done')
    df = pd.DataFrame()
    features_name_list, importance_mean_list, importance_std_list = [], [], []
    for i in r.importances_mean.argsort()[::-1]:
        features_name_list.append(feature_names[i])
        importance_mean_list.append(r.importances_mean[i])
        importance_std_list.append(r.importances_std[i])
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{feature_names[i]:<8}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")
    df['features_name'] = features_name_list
    df['feature_importance'] = importance_mean_list
    df['importance_std'] = importance_std_list
    df.to_csv(save_path, index=False)
