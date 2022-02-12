import yaml
import pickle
import argparse
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from src.data.prepare import prepare_data, get_crop_type_x_y_pos
from src.model.crop_type import get_model_and_params_dict_grid, get_best_model_initial, test, predict, \
    sample_unlabeled_idx
from src.model.util import get_pipeline, get_addtional_params
from src.utils.logger import get_log_dir, get_logger
from src.utils.scv import construct_grid_to_fold, ModifiedBlockCV, ModifiedSKCV
from src.utils.util import save_cv_results
from src.evaluation.visualize import visualize_cv_fold, visualize_cv_polygons


def classifier_crop_type(args):
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

    kullu_dir = img_dir + '43SFR/raster/' if not testing else img_dir + '43SFR/raster_sample/'
    shimla_dir = img_dir + '43RGQ/raster/' if not testing else img_dir + '43RGQ/raster_sample/'

    # load pretrained model
    estimator = pickle.load(open(f'model/{pretrained}.pkl', 'rb'))

    # prepare train and validation dataset
    df_kullu, meta_kullu, feature_names, polygons_list_kullu = \
        prepare_data(logger=logger, dataset='train_val', feature_dir=kullu_dir, window=None,
                        label_path='./data/ground_truth/apples/kullu.shp', smooth=smooth,
                        engineer_feature=engineer_feature, scaling=scaling, new_bands_name=new_bands_name,
                        fill_missing=fill_missing, check_missing=check_missing,
                        vis_stack=args.vis_stack, vis_profile=args.vis_profile, vis_profile_type='apple')
    x_kullu = df_kullu.loc[:, feature_names]
    df_kullu['cropland_pred'] = estimator.predict(x_kullu)
    
    # df_shimla, meta_shimla, _, polygons_list_shimla = \
    #     prepare_data(logger=logger, dataset='train_val', feature_dir=shimla_dir, window=None,
    #                     label_path='./data/ground_truth/apples/shimla.shp', smooth=smooth,
    #                     engineer_feature=engineer_feature, scaling=scaling, new_bands_name=new_bands_name,
    #                     fill_missing=fill_missing, check_missing=check_missing,
    #                     vis_stack=args.vis_stack, vis_profile=args.vis_profile, vis_profile_type='apple')

    n_feature = len(feature_names)
    df_pos_kullu, x_pos_kullu, y_pos_kullu = \
        get_crop_type_x_y_pos(logger, df=df_kullu, n_feature=n_feature, dataset='train_val_kullu')
    # df_train_val_shimla, x_train_val_shimla, y_train_val_shimla = \
    #     get_crop_type_x_y_pos(logger, df=df_shimla, n_feature=n_feature, dataset='train_val_shimla')
    # df_train_val = pd.concat([df_train_val_kullu, df_train_val_shimla], axis=0)
    # x_train_val = np.concatenate((x_train_val_kullu, x_train_val_shimla), axis=0)
    # y_train_val = np.concatenate((y_train_val_kullu, y_train_val_shimla), axis=0)
    # polygons = polygons_list_kullu + polygons_list_shimla
    
    df_pos, x_pos, y_pos, polygons = df_pos_kullu, x_pos_kullu, y_pos_kullu, polygons_list_kullu
    coords_pos = gpd.GeoDataFrame({'geometry': df_pos.coords.values})

    # cross validation
    if cv_type == 'random':
        cv = n_fold
    elif cv_type == 'block':
        # assign to fold
        grid = construct_grid_to_fold(polygons, tiles_x=tiles_x, tiles_y=tiles_y, shape=shape,
                                      data=x_pos, n_fold=n_fold, random_state=random_state)
        scv = ModifiedBlockCV(custom_polygons=grid, buffer_radius=buffer_radius)
        # vis
        if args.vis_cv:
            cv_name = f'./figs/croptype_cv_{tiles_x}x{tiles_y}{shape}_f{n_fold}_s{random_state}'
            visualize_cv_fold(grid, meta_kullu, cv_name + '.tiff')
            logger.info(f'Saved {cv_name}.tiff')
            visualize_cv_polygons(scv, coords_pos, meta_kullu, cv_name + '_mask.tiff')
            logger.info(f'Saved {cv_name}_mask.tiff')
    elif cv_type == 'spatial':
        scv = ModifiedSKCV(n_splits=n_fold, buffer_radius=buffer_radius, random_state=random_state)
        if args.vis_cv:
            cv_name = f'./figs/croptype_cv_{cv_type}_f{n_fold}_s{random_state}'
            visualize_cv_polygons(scv, coords_pos, meta_kullu, cv_name + '_mask.tiff')
            logger.info(f'Saved {cv_name}_mask.tiff')

    # pre-defined parameters
    predefined_params = {
        'ocsvc': {'classification__kernel': 'poly', 'classification__gamma': 'auto', 'classification__nu': 0.5},
        'pul': {'classification__estimator': RandomForestClassifier(), 'classification__hold_out_ratio': 0.1},
        'pul-w': {'classification__estimator': RandomForestClassifier(), 'classification__hold_out_ratio': 0.1,
                  'classification__labeled': 10, 'classification__unlabeled': 15}
    }

    if train_from == 'cropland':
        # sample unlabeled data
        unl_idx = sample_unlabeled_idx(df_kullu.coords, grid, x_pos.shape[0], meta_kullu)
        df_unl_kullu = df_kullu.loc[unl_idx, :]
        coords_unl_kullu = gpd.GeoDataFrame({'geometry': df_unl_kullu.loc[unl_idx, 'coords'].values})
        x_unl_kullu = df_kullu.loc[unl_idx, feature_names].values
        y_unl_kullu = np.zeros(unl_idx.shape[0], dtype=int)
        # concatenate
        df_pu_kullu = pd.concat([df_pos_kullu, df_unl_kullu], axis=0)
        x_pu_kullu = np.concatenate((x_pos_kullu, x_unl_kullu), axis=0)
        y_pu_kullu = np.concatenate((y_pos_kullu, y_unl_kullu), axis=0)
        coords_pu_kullu = gpd.GeoDataFrame(pd.concat([coords_pos, coords_unl_kullu], axis=0))
        for model_name in models_name:
            logger.info(f'## {model_name.upper()}')

            # grid search
            if cv_type:
                # get model and parameters
                model, params_dict_grid = get_model_and_params_dict_grid(model_name, testing)
                params_grid = get_addtional_params(params_dict_grid, testing, study_scaling, engineer_feature)
                logger.info(f'Grid parameters dict {params_grid}')
                if cv_type == 'block' or cv_type == 'spatial':
                    if model_name == 'ocsvm':
                        cv = scv.split(coords_pos)
                    else:  # model_name == 'pul' or 'pul-w'
                        cv = scv.split(coords_pu_kullu)
                # build pipeline
                pipe = get_pipeline(model, scaling, study_scaling, engineer_feature)
                logger.info(pipe)
                # search hyperparameters
                search = GridSearchCV(estimator=pipe, param_grid=params_grid, scoring='recall',
                                      cv=cv, verbose=3, n_jobs=-1)
                if model_name == 'ocsvm':
                    search.fit(x_pos_kullu, y_pos_kullu)
                else:
                    search.fit(x_pu_kullu, y_pu_kullu)
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
                if model_name == 'ocsvm':
                    best_estimator.fit(x_pos_kullu, y_pos_kullu)
                else:
                    best_estimator.fit(x_pu_kullu, y_pu_kullu)
            pickle.dump(best_estimator, open(f'model/{log_time}_{model_name}.pkl', 'wb'))

            # predict and evaluation
            test(logger, best_estimator, x_pu_kullu, y_pu_kullu, meta_kullu, df_pu_kullu.index,
                 pred_name=f'{log_time}_{model_name}_kullu_labels',
                 region_indicator='./data/ground_truth/apples/kullu.shp', color_by_height=color_by_height)
            # test(logger, best_estimator, x_train_val_shimla, y_train_val_shimla, meta_shimla, df_train_val_shimla.index,
            #      pred_name=f'{log_time}_{model_name}_shimla_labels',
            #      region_indicator='./data/ground_truth/apples/kullu.shp', color_by_height=color_by_height)
            
            if not predict_labels_only:
                predict(logger, best_estimator, x_kullu, meta_kullu,
                        pred_path=f'./preds/{log_time}_{model_name}_kullu.tiff', ancillary_dir=ancillary_dir,
                        region_indicator='./data/ground_truth/apples/kullu.shp',
                        color_by_height=color_by_height)
                # x_shimla = df_shimla.loc[:, feature_names]
                # predict(logger, best_estimator, x_shimla, meta_shimla,
                #         pred_path=f'./preds/{log_time}_{model_name}_shimla.tiff', ancillary_dir=ancillary_dir,
                #         region_indicator='./data/ground_truth/apples/shimla.shp',
                #         color_by_height=color_by_height)


    else:  # train_from == 'scratch'
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str,
                        default='./data/config/crop_type_workstation.yaml')
    parser.add_argument('--vis_stack', type=bool, default=False)
    parser.add_argument('--vis_profile', type=bool, default=True)
    parser.add_argument('--vis_cv', type=bool, default=True)

    args = parser.parse_args()

    classifier_crop_type(args)
