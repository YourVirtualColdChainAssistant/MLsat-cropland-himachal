import os
import yaml
import copy
import glob
import pickle
import datetime
import argparse
import numpy as np
import multiprocessing
from rasterio.windows import Window
from rasterio.merge import merge

from src.data.load import clean_test_shapefiles
from src.data.prepare import prepare_data, get_valid_cropland_x_y
from src.utils.logger import get_log_dir, get_logger
from src.models.cropland import test, predict


def cropland_predict(args):
    # read configure file
    with open(args.config_filename) as f:
        config = yaml.load(f)
    data_kwargs = config.get('data')
    model_kwargs = config.get('model')
    predict_kwargs = config.get('predict')
    # data path kwargs
    img_dir = data_kwargs.get('img_dir')
    ancillary_dir = data_kwargs.get('ancillary_dir')
    # model kwargs
    fill_missing = model_kwargs.get('fill_missing')
    check_missing = model_kwargs.get('check_missing')
    scaling = model_kwargs.get('scaling')
    engineer_feature = model_kwargs.get('engineer_feature')
    new_bands_name = model_kwargs.get('new_bands_name')
    smooth = model_kwargs.get('smooth')
    pretrained = model_kwargs.get('pretrained')
    # predict kwargs
    color_by_height = predict_kwargs.get('color_by_height')

    testing = False

    # logger
    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f'cropland_{log_time}_predict.log' if not testing else f'cropland_testing_{log_time}_predict.log'
    logger = get_logger(get_log_dir('./logs/'), __name__, log_filename, level='INFO')
    logger.info(config)

    logger.info('#### Test Cropland Model')
    test_near_dir = img_dir + '43SFR/raster/' if not testing else img_dir + '43SFR/raster_sample/'
    test_far_dir = img_dir + '43RGQ/raster/' if not testing else img_dir + '43RGQ/raster_sample/'
    predict_dir = img_dir + args.tile_id + '/raster/' if not testing else img_dir + args.tile_id + '/raster_sample/'

    if not args.test_regions:
        # load pretrained model
        logger.info("Loading the best pretrained model...")
        best_estimator = pickle.load(open(f'./models/{pretrained}.pkl', 'rb'))

        # read and predict in patches
        full_len, n_patch = 10800, 10
        patch_len = int(full_len / n_patch)

        # check path existence
        pred_path_top = f'{img_dir}predictions/{args.tile_id}/'
        if not os.path.isdir(pred_path_top):
            os.makedirs(pred_path_top)

        # predict patch by patch
        for row in np.linspace(0, full_len, n_patch, endpoint=False, dtype=int):
            for col in np.linspace(0, full_len, n_patch, endpoint=False, dtype=int):
                # get window
                window = Window(col, row, patch_len, patch_len)  # (col_off, row_off, width, height)
                logger.info(f'==== Preparing for {window} ====')

                # prepare data
                df, meta, feature_names = \
                    prepare_data(logger=logger, dataset='predict', feature_dir=predict_dir,
                                 label_path=None, window=window, smooth=smooth,
                                 engineer_feature=engineer_feature, scaling=scaling, new_bands_name=new_bands_name,
                                 fill_missing=fill_missing, check_missing=check_missing,
                                 vis_stack=args.vis_stack, vis_profile=args.vis_profile, vis_profile_type='cropland')
                # get x data
                cat_mask = df.cat_mask.values
                x = df.loc[:, feature_names]
                logger.info(f'df.shape {df.shape}, x.shape {x.shape}')
                # predict
                pred_path = pred_path_top + str(row) + '_' + str(col) + '.tiff'
                predict(logger, best_estimator, x, meta, cat_mask,
                        pred_path=pred_path, ancillary_dir=ancillary_dir,
                        color_by_height=color_by_height, region_indicator=window, eval_open=False)

        # merge patches into single raster
        logger.info('Merging patches...')
        patches_list = [f for f in glob.glob(pred_path_top + '*.tiff')]
        merge(patches_list, dst_path=pred_path_top + args.tile_id + '.tiff')
    else:
        if testing:
            test_dir_dict = {'kullu': test_near_dir}
        else:
            test_dir_dict = {'kullu': test_near_dir, 'mandi': test_far_dir, 'shimla': test_far_dir}
            # test_dir_dict = {'shimla': test_far_dir}
        clean_test_shapefiles()

        pretrained = [pretrained] if isinstance(pretrained, str) else pretrained
        for district in test_dir_dict.keys():
            logger.info(f'### Test on {district}')
            test_dir = test_dir_dict[district]
            label_path = f'./data/ground_truth/test_labels_{district}/test_labels_{district}.shp'
            # prepare data
            df_te, meta, feature_names, _, _ = \
                prepare_data(logger=logger, dataset=f'test_{district}', feature_dir=test_dir,
                             label_path=label_path, window=None, smooth=smooth,
                             engineer_feature=engineer_feature, scaling=scaling, new_bands_name=new_bands_name,
                             fill_missing=fill_missing, check_missing=check_missing,
                             vis_stack=args.vis_stack, vis_profile=args.vis_profile, vis_profile_type='cropland')
            n_feature = len(feature_names)
            cat_mask = df_te.cat_mask.values
            df_test, x_test, y_test = \
                get_valid_cropland_x_y(logger, df=df_te, n_feature=n_feature, dataset=f'test_{district}')
            # test
            for p in pretrained:
                best_estimator = pickle.load(open(f'./models/{p}.pkl', 'rb'))
                test(logger, best_estimator, x_test, y_test, meta, index=df_test.index, cat_mask=cat_mask,
                     pred_name=f'{p}_{district}', ancillary_dir=ancillary_dir, feature_names=None,
                     region_indicator=label_path, color_by_height=color_by_height)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str,
                        default='./data/config/cropland_workstation.yaml')

    parser.add_argument('--tile_ids', nargs='+', default=['43RGQ'])
    parser.add_argument('--test_regions', type=bool, default=True)
    parser.add_argument('--vis_stack', type=bool, default=False)
    parser.add_argument('--vis_profile', type=bool, default=False)

    args = parser.parse_args()

    args_list, tile_ids = [], args.tile_ids
    print(f'Parallizing to {len(tile_ids)} processes...')
    for tile_id in tile_ids:
        args.tile_id = tile_id
        args_list.append(copy.deepcopy(args))  # deep copy 
    process_pool = multiprocessing.Pool(processes=len(tile_ids))
    process_pool.map(cropland_predict, args_list)
