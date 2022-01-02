import sys
import os
import argparse
import datetime
# import skgstat as skg
import numpy as np
import geopandas as gpd

from src.data.prepare import prepare_data, construct_grid_to_fold, clean_train_shapefiles, get_valid_cropland_x_y
from src.utils.logger import get_log_dir, get_logger
from src.utils.scv import ModifiedBlockCV, ModifiedSKCV
from src.evaluation.visualize import visualize_cv_fold, visualize_cv_polygons

from src.models.cropland import get_model_and_params_dict_grid, get_best_model_initial, get_pipeline, test, predict
from sklearn.model_selection import GridSearchCV
from src.utils.util import save_cv_results
import pickle
import yaml


def cropland_classification(args):
    # read configure file
    with open(args.config_filename) as f:
        config = yaml.load(f)
    data_kwargs = config.get('data')
    model_kwargs = config.get('model')
    train_kwargs = config.get('train')
    img_dir = data_kwargs.get('img_dir')

    testing = True
    # logger
    log_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
    log_filename = f'cropland_{log_time}.log' if not testing else f'cropland_testing_{log_time}.log'
    logger = get_logger(get_log_dir('./logs/'), __name__, log_filename, level='INFO')
    logger.info(config)

    logger.info('#### Cropland Classification')
    clean_train_shapefiles()
    feature_dir = img_dir + '/43SFR/raster/' if not testing else img_dir + '/43SFR/raster_sample/'

    # prepare train and validation dataset
    df_tv, meta, feature_names, polygons, _ = \
        prepare_data(logger=logger, dataset='train_val', feature_dir=feature_dir, window=None,
                     label_path='./data/train_labels/train_labels.shp',
                     engineer_feature=args.engineer_feature,
                     scaling=args.scaling, smooth=args.smooth,
                     fill_missing=args.fill_missing, check_missing=args.check_missing,
                     vis_stack=args.vis_stack, vis_profile=args.vis_profile)
    n_feature = len(feature_names)
    cat_mask = df_tv.cat_mask.values 
    df_train_val, x_train_val, y_train_val = \
        get_valid_cropland_x_y(logger, df=df_tv, n_feature=n_feature, dataset='train_val')
    coords_train_val = gpd.GeoDataFrame({'geometry': df_train_val.coords.values})

    # cross validation
    if args.cv_type == 'random':
        cv = args.n_fold
    elif args.cv_type == 'block':
        # assign to fold
        grid = construct_grid_to_fold(polygons, tiles_x=args.tiles_x, tiles_y=args.tiles_y, shape=args.shape,
                                      data=x_train_val, n_fold=args.n_fold, random_state=args.random_state)
        scv = ModifiedBlockCV(custom_polygons=grid, buffer_radius=args.buffer_radius)
        if args.vis_cv:
            cv_name = f'./figs/cv_{args.tiles_x}x{args.tiles_y}{args.shape}_f{args.n_fold}_s{args.random_state}'
            visualize_cv_fold(grid, meta, cv_name + '.tiff')
            logger.info(f'Saved {cv_name}.tiff')
            visualize_cv_polygons(scv, coords_train_val, meta, cv_name + '_mask.tiff')
            logger.info(f'Saved {cv_name}_mask.tiff')
    elif args.cv_type == 'spatial':
        scv = ModifiedSKCV(n_splits=args.n_fold, buffer_radius=args.buffer_radius, random_state=args.random_state)
        if args.vis_cv:
            cv_name = f'./figs/cv_{args.cv_type}_f{args.n_fold}_s{args.random_state}'
            visualize_cv_polygons(scv, coords_train_val, meta, cv_name + '_mask.tiff')
            logger.info(f'Saved {cv_name}_mask.tiff')

    # pre-defined parameters
    predefined_params = {
        'svc': {'classification__C': 10, 'classification__gamma': 'auto', 
                'classification__kernel': 'poly', 'classification__random_state': args.random_state},
        'rfc': {'classification__criterion': 'entropy', 'classification__max_depth': 10, 'classification__max_samples': 0.8, 
                'classification__n_estimators': 100, 'classification__random_state': args.random_state},
        'mlp': {'classification__activation': 'relu', 'classification__alpha': 0.0001, 
                'classification__early_stopping': True, 'classification__hidden_layer_sizes': (300,), 
                'classification__max_iter': 200, 'classification__random_state': args.random_state}
    }

    # pipeline
    for model_name in args.models_mame:
        logger.info(f'## {model_name.upper()}')

        # grid search
        if args.cv_type:
            # get model and parameters
            model, params_grid = get_model_and_params_dict_grid(model_name, args.random_state, testing, args.study_scaling)
            logger.info(f'Grid parameters dict {params_grid}')
            if args.cv_type == 'block' or args.cv_type == 'spatial':
                cv = scv.split(coords_train_val)
            # build pipeline
            pipe = get_pipeline(model, args.scaling, study_scaling=args.study_scaling, engineer_feature=args.engineer_feature)
            logger.info(pipe)
            # search hyperparameters
            search = GridSearchCV(estimator=pipe, param_grid=params_grid, cv=cv, verbose=3, n_jobs=-1)
            search.fit(x_train_val, y_train_val)
            # best parameters
            logger.info(f"Best score {search.best_score_:.4f} with best parameters: {search.best_params_}")
            # cross-validation result
            save_cv_results(search.cv_results_, f'./results/{log_time}_cv_{model_name}.csv')
            best_estimator = search.best_estimator_
        else:
            best_model_init = get_best_model_initial(model_name, predefined_params[model_name])
            if args.engineer_feature == 'select':
                raise ValueError('Cannot build pipeline with SequentialFeatureSelector while using predefined parameters.')
            best_estimator = get_pipeline(best_model_init, args.scaling, 
                                            study_scaling=False, engineer_feature=args.engineer_feature)
            logger.info(pipe)
            best_estimator.fit(x_train_val, y_train_val)
        pickle.dump(best_estimator, open(f'./models/{log_time}_{model_name}.pkl', 'wb'))

        # predict and evaluation
        test(logger, best_estimator, x_train_val, y_train_val, meta, df_train_val.index, 
             cat_mask=cat_mask, region_shp_path='./data/train_labels/train_labels.shp', color_by_height=args.color_by_height, 
             pred_name=f'{log_time}_{model_name}_labels', feature_names=None, work_station=args.work_station)
        if args.predict_train:
            x = df_tv.loc[:, feature_names]
            predict(logger, best_estimator, x, meta, cat_mask=cat_mask,
                    region_shp_path='./data/train_region/train_region.shp', color_by_height=args.color_by_height, 
                    pred_name=f'{log_time}_{model_name}', work_station=args.work_station)
    # if args.check_SAC:
    #     # TODO: draw more pairs below 5km, see the values of auto-correlation
    #     # TODO: how many pixels are we moving if use buffer_radius=3km as suggested in the semi-variogram
    #     # visualize autocorrelation
    #     sample_ids_equal = np.linspace(0, df_train_val.shape[0], num=2000, dtype=int, endpoint=False)
    #     sample_ids_near = np.arange(2000)
    #     # equal
    #     coords_xy = np.array([[coord.x, coord.y] for coord in df_train_val.coords.values])
    #     variogram_equal = skg.Variogram(coords_xy[sample_ids_equal], df_train_val.gt_cropland.values[sample_ids_equal])
    #     variogram_equal.plot().savefig('./figs/semivariogram_equal.png', bbox_inches='tight')
    #     # near
    #     variogram_near = skg.Variogram(coords_xy[sample_ids_near], df_train_val.gt_cropland.values[sample_ids_near])
    #     variogram_near.plot().savefig('./figs/semivariogram_near.png', bbox_inches='tight')
    #     logger.info('Saved semivariogram')

    # TODO: how to use semi-variogram or other statistics in practice
    # TODO: why to study the effects of buffer radius
    # TODO: add NoData mask when predicting for pixels with only missing data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: put all path into yaml file 
    # TODO: add a configuration file and save with the same file name
    # parser.add_argument('--work_station', type=bool, default=True)
    parser.add_argument('--config_filename', type=str, default='./data/config/cropland.yaml')
    
    # cross validation
    # parser.add_argument('--cv_type', default=None, choices=[None, 'random', 'block', 'spatial'],
    #                     help='Method of cross validation.')
    # parser.add_argument('--tiles_x', type=int, default=4)
    # parser.add_argument('--tiles_y', type=int, default=4)
    # parser.add_argument('--shape', type=str, default='square')
    # parser.add_argument('--buffer_radius', type=int, default=0)  # TODO: buffer changes to meter
    # parser.add_argument('--n_fold', type=int, default=3)
    # parser.add_argument('--random_state', type=int, default=24)
    # parser.add_argument('--hp_search_by', type=str, default='grid', choices=['random', 'grid'],
    #                     help='Method to find hyper-parameters.')

    # visualize
    parser.add_argument('--vis_stack', type=bool, default=False)
    parser.add_argument('--vis_profile', type=bool, default=False)
    parser.add_argument('--vis_cv', type=bool, default=False)

    # model
    # parser.add_argument('--engineer_feature', default='temporal', 
    #                     choices=[None, 'temporal', 'temporal+spatial', 'select'] )
    # parser.add_argument('--smooth', type=bool, default=False)
    # parser.add_argument('--study_scaling', type=bool, default=False)
    # parser.add_argument('--scaling', type=str, default='as_reflectance',
    #                     choices=['as_float', 'as_reflectance', 'standardize', 'normalize'])
    # parser.add_argument('--fill_missing', default=None, choices=[None, 'forward', 'linear'])
    # parser.add_argument('--check_missing', type=bool, default=False)
    # parser.add_argument('--check_SAC', type=bool, default=False) 
    # parser.add_argument('--models_mame', nargs="+", default=['rfc', 'svc', 'mlp'])
    
    # parser.add_argument('--predict_train', type=bool, default=True)
    # parser.add_argument('--color_by_height', type=bool, default=True)
    args = parser.parse_args()

    cropland_classification(args)
