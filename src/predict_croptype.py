import os
import yaml
import copy
import glob
import pickle
import datetime
import argparse
import numpy as np
import pandas as pd
import multiprocessing
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.merge import merge

from src.data.load import clean_test_shapefiles, clean_random_shapefile
from src.data.prepare import prepare_data, get_crop_type_x_y_pos, get_unlabeled_pixels
from src.utils.logger import get_log_dir, get_logger
from src.model.crop_type import test, predict #, evaluate_by_feature_importance


def croptype_predict(args):
    # read configure file
    with open(args.config_filename) as f:
        config = yaml.load(f)
    data_kwargs = config.get('data')
    model_kwargs = config.get('model')
    train_kwargs = config.get('train')
    predict_kwargs = config.get('predict')
    # data path kwargs
    img_dir = data_kwargs.get('img_dir')
    ancillary_dir = data_kwargs.get('ancillary_dir')
    # train kwargs
    cv_type = train_kwargs.get('cv_type')
    tiles_x = train_kwargs.get('tiles_x')
    tiles_y = train_kwargs.get('tiles_y')
    shape = train_kwargs.get('shape')
    buffer_radius = train_kwargs.get('buffer_radius')
    n_fold = train_kwargs.get('n_fold')
    random_state = train_kwargs.get('random_state')
    hp_search_by = train_kwargs.get('hp_search_by')
    train_from = train_kwargs.get('train_from')
    # model kwargs
    fill_missing = model_kwargs.get('fill_missing')
    check_missing = model_kwargs.get('check_missing')
    scaling = model_kwargs.get('scaling')
    study_scaling = model_kwargs.get('study_scaling')
    engineer_feature = model_kwargs.get('engineer_feature')
    new_bands_name = model_kwargs.get('new_bands_name')
    smooth = model_kwargs.get('smooth')
    models_name = model_kwargs.get('models_name')
    pretrained = model_kwargs.get('pretrained')
    # predict kwargs
    predict_labels_only = predict_kwargs.get('predict_labels_only')
    color_by_height = predict_kwargs.get('color_by_height')

    testing = False
    

    # logger
    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    log_filename = f'crop_type_{log_time}_on_{pretrained}.log' if not testing else f'crop_type_testing_{log_time}_on_{pretrained}.log'
    logger = get_logger(get_log_dir('./logs/'), __name__, log_filename, level='INFO')
    logger.info(config)


    logger.info('#### Test Crop Type Model')
    test_near_dir = img_dir + '43SFR/raster/' if not testing else img_dir + '43SFR/raster_sample/'
    test_far_dir = img_dir + '43RGQ/raster/' if not testing else img_dir + '43RGQ/raster_sample/'
    predict_dir = img_dir + args.tile_id + '/raster/' if not testing else img_dir + args.tile_id + '/raster_sample/'
    

    if args.action == 'test_from_cropland':
        
        label_path = f'./data/ground_truth/test_labels_combined/polygons_surveys_20210716_20210825_20211213_20220103.shp'
        mandi_path, shimla_path, kullu_path = clean_random_shapefile(label_path)
        label_path_dict = {'kullu': kullu_path}#, 'mandi': mandi_path, 'shimla': shimla_path}

        test_dir_dict = {'kullu': test_near_dir} #, 'mandi': test_far_dir, 'shimla': test_far_dir}

        # load pretrained model
        logger.info("Loading the best pretrained model...")
        best_estimator = pickle.load(open(f'model/{pretrained}.pkl', 'rb'))

        # check path existence to save predictions
        pred_path_top =  f'preds/' #f'{img_dir}predictions/{pretrained}/{args.tile_id}/'
        if not os.path.isdir(pred_path_top):
            os.makedirs(pred_path_top)

        # predict district by district
        for district in test_dir_dict.keys():
            logger.info(f'### Test on {district}')
            test_dir = test_dir_dict[district]
            
            # prepare data
            df_te, meta, feature_names, polygons_list = \
                prepare_data(logger=logger, dataset=f'test_{district}', feature_dir=test_dir,
                                label_path=label_path_dict[district], window=None, smooth=smooth,
                                engineer_feature=engineer_feature, scaling=scaling, new_bands_name=new_bands_name,
                                fill_missing=fill_missing, check_missing=check_missing,
                                vis_stack=args.vis_stack, vis_profile=args.vis_profile, vis_profile_type='cropland',
                                vis_afterprocess=args.vis_afterprocess)
            n_feature = len(feature_names)

            # get cropland masked data
            df_masked, x_masked, y_masked = \
                get_valid_cropland_x_y(logger, df_te, n_feature=n_feature, dataset='from_cropland')

            # get positive samples
            df_pos, x_pos, y_pos = \
                get_crop_type_x_y_pos(logger, df=df_masked, n_feature=n_feature, dataset='from_cropland')

            # get unlabeled samples from whole area
            df_unl, x_unl, y_unl = \
                get_unlabeled_pixels(logger, df=df_te, size=None, dataset='from_cropland')


            # concatenate data
            df_pu = pd.concat([df_pos, df_unl], axis=0).sample(frac=1) #shuffle rows
            x_pu = np.concatenate((x_pos, x_unl), axis=0)
            y_pu = np.concatenate((y_pos, y_unl), axis=0)
                
            test(logger, best_estimator, x_pu, y_pu, meta, df_pu.index,
                 pred_name=f'{district}_from_cropland', color_by_height=color_by_height, region_indicator=label_path)


        # merge patches into single raster
        logger.info('Merging districts...')
        patches_list = [f for f in glob.glob(pred_path_top + '*.tiff')]
        merge(patches_list, dst_path=pred_path_top + '../' + pretrained + '_test_from_cropland.tiff')

    elif args.action == 'test_from_scratch':

        label_path = f'./data/ground_truth/test_labels_combined/polygons_surveys_20210716_20210825_20211213_20220103.shp'
        mandi_path, shimla_path, kullu_path = clean_random_shapefile(label_path)
        label_path_dict = {'kullu': kullu_path}#, 'mandi': mandi_path, 'shimla': shimla_path}

        test_dir_dict = {'kullu': test_near_dir} #, 'mandi': test_far_dir, 'shimla': test_far_dir}

        # load pretrained model
        logger.info("Loading the best pretrained model...")
        best_estimator = pickle.load(open(f'model/{pretrained}.pkl', 'rb'))

        # check path existence to save predictions
        pred_path_top =  f'preds/' #f'{img_dir}predictions/{pretrained}/{args.tile_id}/'
        if not os.path.isdir(pred_path_top):
            os.makedirs(pred_path_top)


        # predict district by district
        for district in test_dir_dict.keys():
            logger.info(f'### Test on {district}')
            test_dir = test_dir_dict[district]
            
            # prepare data
            df_te, meta, feature_names, polygons_list = \
                prepare_data(logger=logger, dataset=f'test_{district}', feature_dir=test_dir,
                                label_path=label_path_dict[district], window=None, smooth=smooth,
                                engineer_feature=engineer_feature, scaling=scaling, new_bands_name=new_bands_name,
                                fill_missing=fill_missing, check_missing=check_missing,
                                vis_stack=args.vis_stack, vis_profile=args.vis_profile, vis_profile_type='cropland',
                                vis_afterprocess=args.vis_afterprocess)
            n_feature = len(feature_names)

            # get positive samples
            df_pos, x_pos, y_pos = \
                get_crop_type_x_y_pos(logger, df=df_te, n_feature=n_feature, dataset='from_scratch')

            # get (all) unlabeled samples
            df_unl, x_unl, y_unl = \
                get_unlabeled_pixels(logger, df=df_te, size=None, n_feature=n_feature, dataset='from_scratch')

            # concatenate data
            df_pu = pd.concat([df_pos, df_unl], axis=0).sample(frac=1) #shuffle rows
            x_pu = np.concatenate((x_pos, x_unl), axis=0)
            y_pu = np.concatenate((y_pos, y_unl), axis=0)
                
            test(logger, best_estimator, x_pu, y_pu, meta, df_pu.index,
                 pred_name=f'{district}_from_scratch', color_by_height=color_by_height, region_indicator=label_path)

        # merge patches into single raster
        logger.info('Merging districts...')
        patches_list = [f for f in glob.glob(pred_path_top + '*.tiff')]
        merge(patches_list, dst_path=pred_path_top + '../' + pretrained + '_test_from_scratch.tiff')

    elif args.action == 'predict_from_scratch':
        # load pretrained model
        logger.info("Loading the best pretrained model...")
        best_estimator = pickle.load(open(f'model/{pretrained}.pkl', 'rb'))

        # read and predict in patches
        with rasterio.open(predict_dir + os.listdir(predict_dir)[0], 'r') as f:
            tile_meta = f.meta
        full_len, n_patch = tile_meta['height'], 10
        patch_len = int(full_len / n_patch)
        logger.info(f'Tile size: {full_len}x{full_len}, num / side: {n_patch}, side size: {patch_len}')

        # check path existence
        pred_path_top =  f'preds/' #f'{img_dir}predictions/{pretrained}/{args.tile_id}/'
        if not os.path.isdir(pred_path_top):
            os.makedirs(pred_path_top)


        # predict patch by patch
        for row in np.linspace(0, full_len, n_patch, endpoint=False, dtype=int):
            for col in np.linspace(0, full_len, n_patch, endpoint=False, dtype=int):
                # get window
                window = Window(col, row, patch_len, patch_len)  # (col_off, row_off, width, height)
                logger.info(f'==== Preparing for {args.tile_id} {window} ====')

                # prepare data
                df, meta, feature_names = \
                    prepare_data(logger=logger, dataset='predict', feature_dir=predict_dir,
                                 label_path=None, window=window, smooth=smooth,
                                 engineer_feature=engineer_feature, scaling=scaling, new_bands_name=new_bands_name,
                                 fill_missing=fill_missing, check_missing=check_missing,
                                 vis_stack=args.vis_stack, vis_profile=args.vis_profile, vis_profile_type='cropland',
                                 vis_afterprocess=args.vis_afterprocess)
                # get x data
                x = df.loc[:, feature_names]
                logger.info(f'df.shape {df.shape}, x.shape {x.shape}')
                # predict
                pred_path = pred_path_top + str(row) + '_' + str(col) + '.tiff'
                predict(logger, best_estimator, x, meta=meta, pred_path=pred_path,
                        color_by_height=color_by_height, region_indicator=window)

        # merge patches into single raster
        logger.info('Merging patches...')
        patches_list = [f for f in glob.glob(pred_path_top + '*.tiff')]
        merge(patches_list, dst_path=pred_path_top + '../' + args.tile_id + '.tiff')
    
    elif args.action == 'predict_from_cropland':
        # load pretrained model
        logger.info("Loading the best pretrained model...")
        best_estimator = pickle.load(open(f'model/{pretrained}.pkl', 'rb'))

        # read and predict in patches
        with rasterio.open(predict_dir + os.listdir(predict_dir)[0], 'r') as f:
            tile_meta = f.meta
        full_len, n_patch = tile_meta['height'], 10
        patch_len = int(full_len / n_patch)
        logger.info(f'Tile size: {full_len}x{full_len}, num / side: {n_patch}, side size: {patch_len}')

        # check path existence
        pred_path_top =  f'preds/' #f'{img_dir}predictions/{pretrained}/{args.tile_id}/'
        if not os.path.isdir(pred_path_top):
            os.makedirs(pred_path_top)

        cropland_path = f'./data/ground_truth/test_labels_combined/polygons_surveys_20210716_20210825_20211213_20220103.shp'
                
        # predict patch by patch
        for row in np.linspace(0, full_len, n_patch, endpoint=False, dtype=int):
            for col in np.linspace(0, full_len, n_patch, endpoint=False, dtype=int):
                # get window
                window = Window(col, row, patch_len, patch_len)  # (col_off, row_off, width, height)
                logger.info(f'==== Preparing for {args.tile_id} {window} ====')

                # prepare data
                df, meta, feature_names = \
                    prepare_data(logger=logger, dataset='predict_from_cropland', feature_dir=predict_dir,
                                 label_path=None, window=window, smooth=smooth,
                                 engineer_feature=engineer_feature, scaling=scaling, new_bands_name=new_bands_name,
                                 fill_missing=fill_missing, check_missing=check_missing,
                                 vis_stack=args.vis_stack, vis_profile=args.vis_profile, vis_profile_type='cropland',
                                 vis_afterprocess=args.vis_afterprocess)
                
                # get cropland data
                logger.info('# Load raw cropland labels')
                polygons_list, labels = load_shp_to_array(cropland_path, meta)
                df['label'] = labels.reshape(-1)
                print('Labels added in df', df.label.unique())
                logger.info('# Convert to cropland and crop labels')
                df['gt_cropland'] = df.label.values.copy()
                df.loc[df.label.values == 1, 'gt_cropland'] = 2
                print('Modified gt values', df.gt_cropland.unique())

                df_masked, x_masked, y_masked = \
                    get_valid_cropland_x_y(logger, df, n_feature=n_feature, dataset='predict_from_cropland')
                
                x = df_masked.loc[:, feature_names]
                logger.info(f'df.shape {df_masked.shape}, x.shape {x_masked.shape}')
                # predict
                pred_path = pred_path_top + str(row) + '_' + str(col) + '.tiff'
                predict(logger, best_estimator, x_masked, meta=meta, pred_path=pred_path,
                        color_by_height=color_by_height, region_indicator=window)

        # merge patches into single raster
        logger.info('Merging patches...')
        patches_list = [f for f in glob.glob(pred_path_top + '*.tiff')]
        merge(patches_list, dst_path=pred_path_top + '../' + args.tile_id + '.tiff')
        
    else:  # args.action == 'test_scratch_together'
        label_path = f'./data/ground_truth/test_labels_combined/polygons_surveys_20210716_20210825_20211213_20220103.shp'
        mandi_path, shimla_path, kullu_path = clean_random_shapefile(label_path)
        label_path_dict = {'kullu': kullu_path}#, 'mandi': mandi_path, 'shimla': shimla_path}

        test_dir_dict = {'kullu': test_near_dir} #, 'mandi': test_far_dir, 'shimla': test_far_dir}

        # load pretrained model
        logger.info("Loading the best pretrained model...")
        best_estimator = pickle.load(open(f'model/{pretrained}.pkl', 'rb'))

        # store all test data in an array
        for district in test_dir_dict.keys():
            logger.info(f'### Loading {district} data')
            test_dir = test_dir_dict[district]
            
            # prepare data
            df_te, meta, feature_names, polygons_list = \
                prepare_data(logger=logger, dataset=f'test_{district}', feature_dir=test_dir,
                             label_path=label_path, window=None, smooth=smooth,
                             engineer_feature=engineer_feature, scaling=scaling, new_bands_name=new_bands_name,
                             fill_missing=fill_missing, check_missing=check_missing,
                             vis_stack=args.vis_stack, vis_profile=args.vis_profile, vis_profile_type='cropland',
                             vis_afterprocess=args.vis_afterprocess)
            
            x_test_d = df_te.loc[:, feature_names]
            y_test_d = df_te['label']
            
            if district == 'kullu': # initiate the concatenation
                x_test = x_test_d
                y_test = y_test_d
            else:
                x_test = np.concatenate((x_test, x_test_d), axis=0)
                y_test = np.concatenate((y_test, y_test_d), axis=0)

        # test    
        logger.info('Evaluating by feature importance...')
        evaluate_by_feature_importance(best_estimator, x_test, y_test,
                                       feature_names, f'{pretrained}_test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str,
                        default='./data/config/crop_type.yaml')

    parser.add_argument('--tile_ids', nargs='+', default=['43SFR'])
    parser.add_argument('--action', type=str, default='test_from_scratch',
                        choices=['test_from_cropland', 'test_from_scratch', 'predict_from_scratch', 'predict_from_cropland', 'test_together'])
    parser.add_argument('--vis_stack', type=bool, default=False)
    parser.add_argument('--vis_profile', type=bool, default=False)
    parser.add_argument('--vis_afterprocess', type=bool, default=False)

    args = parser.parse_args()

    args_list, tile_ids = [], args.tile_ids
    print(f'Parallizing to {len(tile_ids)} processes...')
    for tile_id in tile_ids:
        args.tile_id = tile_id
        args_list.append(copy.deepcopy(args))  # deep copy 
    process_pool = multiprocessing.Pool(processes=len(tile_ids))
    process_pool.map(croptype_predict, args_list)
