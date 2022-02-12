import sys
import os
import yaml
import pickle
import datetime
import argparse
import numpy as np
import geopandas as gpd
import skgstat as skg
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

from src.data.load import clean_train_shapefiles
from src.data.prepare import prepare_data, prepare_HP_data, get_valid_cropland_x_y
from src.utils.logger import get_log_dir, get_logger
from src.utils.util import save_cv_results
from src.utils.scv import ModifiedBlockCV, ModifiedSKCV, construct_grid_to_fold
from src.model.util import get_pipeline, get_addtional_params
from src.model.cropland import get_model_and_params_dict_grid, get_best_model_initial, test, predict
from src.evaluation.visualize import visualize_cv_fold, visualize_cv_polygons


def cropland_classification(args):
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
    sample_HP = train_kwargs.get('sample_HP')
    # model kwargs
    composite_by = model_kwargs.get('composite_by')
    fill_missing = model_kwargs.get('fill_missing')
    check_missing = model_kwargs.get('check_missing')
    scaling = model_kwargs.get('scaling')
    study_scaling = model_kwargs.get('study_scaling')
    engineer_feature = model_kwargs.get('engineer_feature')
    new_bands_name = model_kwargs.get('new_bands_name')
    smooth = model_kwargs.get('smooth')
    check_SAC = model_kwargs.get('check_SAC')
    models_name = model_kwargs.get('models_name')
    # predict kwargs
    predict_labels_only = predict_kwargs.get('predict_labels_only')
    color_by_height = predict_kwargs.get('color_by_height')

    testing = False
    # logger
    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f'cropland_{log_time}.log' if not testing else f'cropland_testing_{log_time}.log'
    logger = get_logger(get_log_dir('./logs/'), __name__, log_filename, level='INFO')
    logger.info(config)

    logger.info('#### Cropland Classification')
    clean_train_shapefiles()
    feature_dir = img_dir + '43SFR/raster/' if not testing else img_dir + '43SFR/raster_sample/'

    # prepare other polygons in state
    # TODO: add more samples distributed in the whole state
    if sample_HP:
        sample_HP_path = './data/ground_truth/sample_HP/sample_HP.shp'
        df_HP, polygons_HP, _ = \
            prepare_HP_data(logger=logger, img_dir=img_dir, sample_HP_path=sample_HP_path, smooth=smooth,
                            engineer_feature=engineer_feature, scaling=scaling, new_bands_name=new_bands_name,
                            fill_missing=fill_missing, check_missing=check_missing, composite_by=composite_by)

    # prepare train and validation dataset
    df_tv, meta, feature_names, polygons = \
        prepare_data(logger=logger, dataset='train_val', feature_dir=feature_dir, window=None,
                     label_path='./data/ground_truth/train_labels/train_labels.shp', smooth=smooth,
                     engineer_feature=engineer_feature, scaling=scaling, new_bands_name=new_bands_name,
                     fill_missing=fill_missing, check_missing=check_missing, composite_by=composite_by,
                     vis_stack=args.vis_stack, vis_profile=args.vis_profile, vis_profile_type='cropland',
                     vis_afterprocess=args.vis_afterprocess)
    n_feature = len(feature_names)
    df_train_val, x_train_val, y_train_val = \
        get_valid_cropland_x_y(logger, df=df_tv, n_feature=n_feature, dataset='train_val')
    coords_train_val = gpd.GeoDataFrame({'geometry': df_train_val.coords.values})

    # cross validation
    if cv_type == 'random':
        cv = n_fold
    elif cv_type == 'block':
        # assign to fold
        grid = construct_grid_to_fold(polygons, tiles_x=tiles_x, tiles_y=tiles_y, shape=shape,
                                      data=x_train_val, n_fold=n_fold, random_state=random_state)
        scv = ModifiedBlockCV(custom_polygons=grid, buffer_radius=buffer_radius)
        if args.vis_cv:
            cv_name = f'./figs/cv_{tiles_x}x{tiles_y}{shape}_f{n_fold}_s{random_state}'
            visualize_cv_fold(grid, meta, cv_name + '.tiff')
            logger.info(f'Saved {cv_name}.tiff')
            visualize_cv_polygons(scv, coords_train_val, meta, cv_name + '_mask.tiff')
            logger.info(f'Saved {cv_name}_mask.tiff')
    elif cv_type == 'spatial':
        scv = ModifiedSKCV(n_splits=n_fold, buffer_radius=buffer_radius, random_state=random_state)
        if args.vis_cv:
            cv_name = f'./figs/cv_{cv_type}_f{n_fold}_s{random_state}'
            visualize_cv_polygons(scv, coords_train_val, meta, cv_name + '_mask.tiff')
            logger.info(f'Saved {cv_name}_mask.tiff')

    # pre-defined parameters
    predefined_params = {
        'svc': {'classification__C': 10, 'classification__gamma': 'auto',
                'classification__kernel': 'poly', 'classification__random_state': random_state},
        'rfc': {'classification__criterion': 'entropy', 'classification__max_depth': 10,
                'classification__max_samples': 0.8,
                'classification__n_estimators': 100, 'classification__random_state': random_state},
        'mlp': {'classification__activation': 'relu', 'classification__alpha': 0.0001,
                'classification__early_stopping': True, 'classification__hidden_layer_sizes': (300,),
                'classification__max_iter': 200, 'classification__random_state': random_state}
    }

    # pipeline
    for model_name in models_name:
        logger.info(f'## {model_name.upper()}')

        # grid search
        if cv_type:
            # get model and parameters
            model, params_dict_grid = get_model_and_params_dict_grid(model_name, random_state, testing)
            params_grid = get_addtional_params(params_dict_grid, testing, study_scaling, engineer_feature)
            logger.info(f'Grid parameters dict {params_grid}')
            if cv_type == 'block' or cv_type == 'spatial':
                cv = scv.split(coords_train_val)
            # build pipeline
            pipe = get_pipeline(model, scaling, study_scaling=study_scaling, engineer_feature=engineer_feature)
            logger.info(pipe)
            # search hyperparameters
            scoring = {'accuracy': make_scorer(accuracy_score), 
                       'precision': make_scorer(precision_score, pos_label=2), 
                       'recall': make_scorer(recall_score, pos_label=2), 
                       'f1_score': make_scorer(f1_score, pos_label=2)}
            search = GridSearchCV(estimator=pipe, param_grid=params_grid, scoring=scoring, refit='accuracy', cv=cv, verbose=3, n_jobs=-1)
            search.fit(x_train_val, y_train_val)
            # best parameters
            logger.info(f"Best score {search.best_score_:.4f} with best parameters: {search.best_params_}")
            # cross-validation result
            save_cv_results(search.cv_results_, f'./results/{log_time}_cv_{model_name}.csv')
            best_estimator = search.best_estimator_
        else:
            best_model_init = get_best_model_initial(model_name, predefined_params[model_name])
            if engineer_feature == 'select':
                raise ValueError(
                    'Cannot build pipeline with SequentialFeatureSelector while using predefined parameters.')
            best_estimator = get_pipeline(best_model_init, scaling,
                                          study_scaling=False, engineer_feature=engineer_feature)
            logger.info(best_estimator)
            best_estimator.fit(x_train_val, y_train_val)
        pickle.dump(best_estimator, open(f'model/{log_time}_{model_name}.pkl', 'wb'))

        # predict and evaluation
        test(logger, best_estimator, x_train_val, y_train_val, meta, df_train_val.index,
             pred_name=f'{log_time}_{model_name}_labels', ancillary_dir=ancillary_dir, feature_names=None,
             region_indicator='./data/ground_truth/train_labels/train_labels.shp', color_by_height=color_by_height)
        if not predict_labels_only:
            x = df_tv.loc[:, feature_names]
            predict(logger, best_estimator, x, meta,
                    pred_path=f'./preds/{log_time}_{model_name}.tiff', ancillary_dir=ancillary_dir,
                    region_indicator='./data/ground_truth/train_region/train_region.shp',
                    color_by_height=color_by_height)

    if check_SAC:
        # TODO: draw more pairs below 5km, see the values of auto-correlation
        # TODO: how many pixels are we moving if use buffer_radius=3km as suggested in the semi-variogram
        # visualize autocorrelation
        sample_ids_equal = np.linspace(0, df_train_val.shape[0], num=2000, dtype=int, endpoint=False)
        sample_ids_near = np.arange(2000)
        # equal
        coords_xy = np.array([[coord.x, coord.y] for coord in df_train_val.coords.values])
        variogram_equal = skg.Variogram(coords_xy[sample_ids_equal], df_train_val.gt_cropland.values[sample_ids_equal])
        variogram_equal.plot().savefig('./figs/semivariogram_equal.png', bbox_inches='tight')
        # near
        variogram_near = skg.Variogram(coords_xy[sample_ids_near], df_train_val.gt_cropland.values[sample_ids_near])
        variogram_near.plot().savefig('./figs/semivariogram_near.png', bbox_inches='tight')
        logger.info('Saved semivariogram')

    # TODO: how to use semi-variogram or other statistics in practice
    # TODO: why to study the effects of buffer radius
    # TODO: add NoData mask when predicting for pixels with only missing data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: buffer radius as meter
    parser.add_argument('--config_filename', type=str, default='./data/config/cropland_workstation.yaml')
    parser.add_argument('--vis_stack', type=bool, default=False)
    parser.add_argument('--vis_profile', type=bool, default=False)
    parser.add_argument('--vis_cv', type=bool, default=False)
    parser.add_argument('--vis_afterprocess', type=bool, default=False)

    args = parser.parse_args()

    cropland_classification(args)
