import os
import fiona
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from sklearn.inspection import permutation_importance
from util import load_geotiff, clip_single_raster

"""
=============================== open datasets ================================
"""


def evaluate_by_open_datasets(args):
    # compare with gfsad
    gfsad_path = args.ancilliary_data + 'cropland/GFSAD30/GFSAD30SAAFGIRCE_2015_N30E70_001_2017286103800.tif'
    gfsad_clip_path = '../data/gfsad_clipped.tiff'
    gfsad_align_path = '../data/gfsad_aligned.tiff'
    pred_path = '../preds/1022-102249_svc.tiff'

    clip_open_datasets_based_on_study_area(gfsad_path, gfsad_clip_path)
    align_raster(pred_path, gfsad_clip_path, gfsad_align_path)
    compare_predictions_with_gfsad(pred_path, gfsad_align_path)

    # compare with copernicus
    copernicus_path = args.ancilliary_data + 'landcover/Copernicus_LC100m/INDIA_2019/' + \
                      'E060N40_PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif'
    copernicus_clip_path = '../data/copernicus_clipped.tiff'
    copernicus_align_path = '../data/copernicus_aligned.tiff'
    pred_path = '../preds/1008-183014_rfc.tiff'

    clip_open_datasets_based_on_study_area(copernicus_path, copernicus_clip_path)
    align_raster(pred_path, copernicus_clip_path, copernicus_align_path)
    compare_predictions_with_copernicus(pred_path, copernicus_align_path)

    # compare predictions
    diff_two_predictions('../preds/1008-183014_rfc.tiff', '../preds/1007-153151_svm.tiff')


def clip_open_datasets_based_on_study_area(input_path, output_path):
    study_area_shp = '../data/study-area/study_area.shp'
    with fiona.open(study_area_shp, "r") as shapefile:
        study_area_shapes = [feature["geometry"] for feature in shapefile if feature["geometry"] is not None]
    study_area_crs = gpd.read_file(study_area_shp).crs
    clip_single_raster(study_area_crs, study_area_shapes, input_path, output_path)


def align_raster(pred_path, input_path, output_path):
    """
    Align according to prediction file (with boundary and resolution adjustment).

    """
    # prepare source info
    bounds = rasterio.open(pred_path).bounds
    _, meta_tar = load_geotiff(pred_path)

    # command
    cmd = f"gdalwarp -overwrite -r average -t_srs {meta_tar['crs']} -ts {meta_tar['width']} {meta_tar['height']} " + \
          f"-te {bounds.left} {bounds.bottom} {bounds.right} {bounds.top} {input_path} {output_path}"
    returned_val = os.system(cmd)
    if returned_val == 0:
        print('Successfully align raster!')
    else:
        print('Alignment failed!')
        exit()


def compare_predictions_with_gfsad(pred_path, dataset_path, logger=None):
    # load data
    band_pred, meta_pred = load_geotiff(pred_path)
    band_dataset, meta_dataset = load_geotiff(dataset_path)
    band_pred = band_pred[0]
    band_dataset = band_dataset[0]

    # rescale to make target value taking 1
    band_pred = band_pred / band_pred.min()
    band_dataset = band_dataset / band_dataset.max()

    # calculate
    num_in_dataset = (band_dataset == 1.0).sum()
    num_in_pred = (band_pred[band_dataset == 1.0] == 1.0).sum()
    if logger == None:
        print(f'Cropland pixel number in GFASD: {num_in_dataset}')
        print(f'Cropland pixel number in prediction: {num_in_pred}')
        print(f'Percentage: {num_in_pred / num_in_dataset * 100:.2f}%')
    else:
        logger.info(f'Cropland pixel number in GFASD: {num_in_dataset}')
        logger.info(f'Cropland pixel number in prediction: {num_in_pred}')
        logger.info(f'Percentage: {num_in_pred / num_in_dataset * 100:.2f}%')


def compare_predictions_with_copernicus(pred_path, dataset_path, logger=None):
    # load data
    band_pred, meta_pred = load_geotiff(pred_path)
    band_dataset, meta_dataset = load_geotiff(dataset_path)
    band_pred = band_pred[0]
    band_dataset = band_dataset[0]

    # rescale to make target value taking 1
    band_pred = band_pred / band_pred.max()
    band_dataset = band_dataset * 255
    band_dataset[(band_dataset == 50) | (band_dataset == 111)] = 1

    # calculate
    num_in_dataset = (band_dataset == 1.0).sum()
    num_in_pred = (band_pred[band_dataset == 1.0] == 1.0).sum()
    if logger is None:
        print(f'Non-cropland pixel number in Copernicus: {num_in_dataset}')
        print(f'Non-cropland pixel number in prediction: {num_in_pred}')
        print(f'Percentage: {num_in_pred / num_in_dataset * 100:.2f}%')
    else:
        logger.info(f'Non-cropland pixel number in Copernicus: {num_in_dataset}')
        logger.info(f'Non-cropland pixel number in prediction: {num_in_pred}')
        logger.info(f'Percentage: {num_in_pred / num_in_dataset * 100:.2f}%')


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
    output_path = f"../preds/diff_{pred_name_1}_{pred_name_2}.tiff"
    # save
    with rasterio.open(output_path, "w", **meta_pred_1) as dst:
        dst.write(band_diff)
        print(f'Save difference between {pred_name_1} and {pred_name_2} to {output_path}')


"""
=============================== feature importance ================================
"""


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ancilliary_data',
        type=str,
        default='K:/2021-data-org/4. RESEARCH_n/ML/MLsatellite/Data/layers_india/ancilliary_data/',
        help='Base directory to all the images.'
    )
    args = parser.parse_args()
    evaluate_by_open_datasets(args)
