import os
import fiona
import argparse
import geopandas as gpd
import rasterio
from util import load_geotiff, clip_single_raster


def validate(args):
    # compare with gfsad
    gfsad_path = args.ancilliary_data + 'cropland/GFSAD30/GFSAD30SAAFGIRCE_2015_N30E70_001_2017286103800.tif'
    gfsad_clip_path = '../data/gfsad_clipped.tiff'
    gfsad_align_path = '../data/gfsad_aligned.tiff'
    pred_path = '../preds/1008-183014_rfc.tif'

    clip_open_datasets_based_on_study_area(gfsad_path, gfsad_clip_path)
    align_raster(pred_path, gfsad_clip_path, gfsad_align_path)

    compare_predictions_with_gfsad(pred_path, gfsad_align_path)

    # compare with copernicus
    copernicus_path = args.ancilliary_data + 'landcover/Copernicus_LC100m/INDIA_2019/' + \
        'E060N40_PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif'
    copernicus_clip_path = '../data/copernicus_clipped.tiff'
    copernicus_align_path = '../data/copernicus_aligned.tiff'
    pred_path = '../preds/1008-183014_rfc.tif'

    clip_open_datasets_based_on_study_area(copernicus_path, copernicus_clip_path)
    align_raster(pred_path, copernicus_clip_path, copernicus_align_path)

    compare_predictions_with_copernicus(pred_path, copernicus_align_path)


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


def compare_predictions_with_gfsad(pred_path, dataset_path):
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
    print(f'Cropland pixel number in GFASD: {num_in_dataset}')
    print(f'Cropland pixel number in prediction: {num_in_pred}')
    print(f'Percentage: {num_in_pred / num_in_dataset * 100:.2f}%')


def compare_predictions_with_copernicus(pred_path, dataset_path):
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
    print(f'Non-cropland pixel number in Copernicus: {num_in_dataset}')
    print(f'Non-cropland pixel number in prediction: {num_in_pred}')
    print(f'Percentage: {num_in_pred / num_in_dataset * 100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ancilliary_data',
        type=str,
        default='K:/2021-data-org/4. RESEARCH_n/ML/MLsatellite/Data/layers_india/ancilliary_data/',
        help='Base directory to all the images.'
    )
    args = parser.parse_args()
    validate(args)
