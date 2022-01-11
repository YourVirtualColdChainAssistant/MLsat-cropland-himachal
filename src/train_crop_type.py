import yaml
import pickle
import argparse
import datetime
import numpy as np
import geopandas as gpd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from src.data.prepare import prepare_data, get_valid_crop_type_x_y
from src.models.crop_type import get_model_and_params_dict_grid, get_best_model_initial, test, predict, \
    sample_unlabeled_idx
from src.models.util import get_pipeline, get_addtional_params
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
    k_feature = model_kwargs.get('k_feature')
    new_bands_name = model_kwargs.get('new_bands_name')
    smooth = model_kwargs.get('smooth')
    models_name = model_kwargs.get('models_name')
    pretrained = model_kwargs.get('pretrained')
    # predict kwargs
    predict_labels_only = predict_kwargs.get('predict_labels_only')
    color_by_height = predict_kwargs.get('color_by_height')

    testing = True
    # logger
    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for p in pretrained:
        log_filename = f'crop_type_{log_time}_on_{p}.log' if not testing else f'crop_type_testing_{log_time}_on_{p}.log'
        logger = get_logger(get_log_dir('./logs/'), __name__, log_filename, level='INFO')
        logger.info(config)

        feature_dir = img_dir + '43SFR/raster/' if not testing else img_dir + '43SFR/raster_sample/'

        # TODO: do twice to load other train region data
        # prepare train and validation dataset
        df_tv, meta, feature_names, polygons, val_list = \
            prepare_data(logger=logger, dataset='train_val', feature_dir=feature_dir, window=None,
                         label_path='./data/ground_truth/train_labels/train_labels.shp', smooth=smooth,
                         engineer_feature=engineer_feature, scaling=scaling, new_bands_name=new_bands_name,
                         fill_missing=fill_missing, check_missing=check_missing,
                         vis_stack=args.vis_stack, vis_profile=args.vis_profile, vis_profile_type='apples')
        n_feature = len(feature_names)
        cat_mask = df_tv.cat_mask.values
        df_train_val, x_train_val, y_train_val = \
            get_valid_crop_type_x_y(logger, df=df_tv, n_feature=n_feature, dataset='train_val')
        coords_train_val = gpd.GeoDataFrame({'geometry': df_train_val.coords.values})

    # cross validation
    if cv_type == 'random':
        cv = n_fold
    elif cv_type == 'block':
        # assign to fold
        # TODO: polygons are inconsistent, could use the property that polygons and val are syncronized 
        polygons_valid = list(np.array(polygons)[np.array(val_list) == 1 or 2])
        grid = construct_grid_to_fold(polygons_valid, tiles_x=tiles_x, tiles_y=tiles_y, shape=shape,
                                      data=x_train_val, n_fold=n_fold, random_state=random_state)
        scv = ModifiedBlockCV(custom_polygons=grid, buffer_radius=buffer_radius)
        if args.vis_cv:
            cv_name = f'./figs/croptype_cv_{tiles_x}x{tiles_y}{shape}_f{n_fold}_s{random_state}'
            visualize_cv_fold(grid, meta, cv_name + '.tiff')
            logger.info(f'Saved {cv_name}.tiff')
            visualize_cv_polygons(scv, coords_train_val, meta, cv_name + '_mask.tiff')
            logger.info(f'Saved {cv_name}_mask.tiff')
    elif cv_type == 'spatial':
        scv = ModifiedSKCV(n_splits=n_fold, buffer_radius=buffer_radius, random_state=random_state)
        if args.vis_cv:
            cv_name = f'./figs/croptype_cv_{cv_type}_f{n_fold}_s{random_state}'
            visualize_cv_polygons(scv, coords_train_val, meta, cv_name + '_mask.tiff')
            logger.info(f'Saved {cv_name}_mask.tiff')

    # pre-defined parameters
    predefined_params = {
        'ocsvc': {'classification__kernel': 'poly', 'classification__gamma': 'auto', 'classification__nu': 0.5},
        'pul': {'classification__estimator': RandomForestClassifier(), 'classification__hold_out_ratio': 0.1},
        'pul-w': {'classification__estimator': RandomForestClassifier(), 'classification__hold_out_ratio': 0.1,
                  'classification__labeled': 10, 'classification__unlabeled': 15}
    }

    if train_from == 'cropland':
        # positive data
        mask_pos = df_train_val.label.values == 1
        x_train_val_pos = x_train_val[mask_pos]
        y_train_val_pos = np.ones(x_train_val_pos.shape[0], type=int)
        # sample unlabeled data
        unl_idx = sample_unlabeled_idx(mask_pos.argmin(axis=0), x_train_val_pos.shape[0])
        x_train_val_unl = x_train_val[unl_idx]
        y_train_val_unl = np.zeros(unl_idx.shape[0], type=int)
        # concat
        x_train_val_pu = np.concatenate([x_train_val_pos, x_train_val_unl], axis=0)
        y_train_val_pu = np.concatenate([y_train_val_pos, y_train_val_unl], axis=0)
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
                pipe = get_pipeline(model, scaling, study_scaling=study_scaling,
                                    engineer_feature=engineer_feature, k_feature=k_feature)
                logger.info(pipe)
                # search hyperparameters
                search = GridSearchCV(estimator=pipe, param_grid=params_grid, scoring='recall',
                                      cv=cv, verbose=3, n_jobs=-1)
                if model_name == 'ocsvm':
                    search.fit(x_train_val_pos, y_train_val_pos)
                else:
                    search.fit(x_train_val_pu, y_train_val_pu)
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
                    best_estimator.fit(x_train_val_pos, y_train_val_pos)
                else:
                    best_estimator.fit(x_train_val_pu, y_train_val_pu)
            pickle.dump(best_estimator, open(f'./models/{log_time}_{model_name}.pkl', 'wb'))

            # predict and evaluation
            # TODO: index is wrong in saving 
            test(logger, best_estimator, x_train_val, y_train_val, meta, df_train_val.index, cat_mask=cat_mask,
                 pred_name=f'{log_time}_{model_name}_labels',
                 region_indicator='./data/ground_truth/train_labels/train_labels.shp',
                 color_by_height=color_by_height)
            if not predict_labels_only:
                x = df_tv.loc[:, feature_names]
                # TODO: lack cropland_mask 
                predict(logger, best_estimator, x, meta, cat_mask=cat_mask,
                        pred_path=f'./preds/{log_time}_{model_name}.tiff',
                        region_indicator='./data/ground_truth/train_region/train_region.shp',
                        color_by_height=color_by_height)
    else:  # train_from == 'scratch'
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str,
                        default='./data/config/crop_type_workstation.yaml')
    parser.add_argument('--vis_stack', type=bool, default=False)
    parser.add_argument('--vis_profile', type=bool, default=False)
    parser.add_argument('--vis_cv', type=bool, default=False)

    args = parser.parse_args()

    classifier_crop_type(args)
