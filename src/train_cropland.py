import argparse
import datetime
from spacv.spacv import SKCV
import geopandas as gpd
from util import get_log_dir, get_logger
from visualization import NVDI_profile, visualize_cv_fold, visualize_cv_polygons
from prepare_data import clean_train_shapefiles, clean_test_shapefiles, prepare_data, construct_grid_to_fold
from models import ModelCropland
from spatial_cv import ModifiedBlockCV


def cropland_classification(args):
    testing = True
    # logger
    log_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
    log_filename = f'cropland_{log_time}.log' if not testing else f'cropland_testing_{log_time}.log'
    logger = get_logger(get_log_dir(), __name__, log_filename, level='INFO')
    logger.info(args)

    logger.info('#### Cropland Classification')
    train_val_dir = args.images_dir + 'train_area/' if not testing else args.images_dir + 'train_area_sample/'
    test_dir = args.images_dir + 'test_region_near/' if not testing else args.images_dir + 'test_region_near_sample/'

    # clean shapefiles
    logger.info('# Clean shapefiles')
    # TODO: fiona.errors.CRSError: Invalid input to create CRS
    # clean_train_shapefiles()
    # clean_test_shapefiles()
    logger.info('  ok')

    # prepare train/validation/test set
    df_tv, df_train_val, x_train_val, y_train_val, polygons, scaler, meta, n_feature, feature_names = \
        prepare_data(logger, dataset='train_val', feature_dir=train_val_dir,
                     label_path='../data/train_labels/train_labels.shp',
                     feature_engineering=args.feature_engineering,
                     feature_scaling=args.feature_scaling,
                     vis_ts=args.vis_ts, vis_profile=args.vis_profile)
    df_te, df_test, x_test, y_test, _, _, _, _, _ = \
        prepare_data(logger, dataset='test', feature_dir=test_dir,
                     label_path='../data/test_labels/test_labels.shp',
                     feature_engineering=args.feature_engineering,
                     feature_scaling=args.feature_scaling, scaler=scaler,
                     vis_ts=args.vis_ts, vis_profile=args.vis_profile)
    logger.info(f'\nFeatures: {feature_names}')
    coords_train_val = gpd.GeoDataFrame({'geometry': df_train_val.coords.values})
    x = df_tv.iloc[:, :n_feature].values
    if args.feature_scaling is not None:
        x = scaler.transform(x)

    # cross validation
    if args.cv_type == 'random':
        cv = args.n_fold
    elif args.cv_type == 'block':
        # assign to fold
        # TODO: change grid assignment, to make sure each fold is more or less balanced
        grid = construct_grid_to_fold(polygons, tiles_x=args.tiles_x, tiles_y=args.tiles_y, shape=args.shape,
                                      data=x_train_val, n_fold=args.n_fold, random_state=args.random_state)
        scv = ModifiedBlockCV(custom_polygons=grid, buffer_radius=args.buffer_radius)
        # visualize valid block
        visualize_cv_fold(grid, meta,
                          save_path=f'../figs/cv_{args.tiles_x}x{args.tiles_y}{args.shape}_{args.n_fold}fold_seed{args.random_state}.tiff')
    else:  # spatial
        scv = SKCV(n_splits=args.n_splits, buffer_radius=args.buffer_radius, random_state=args.random_state)

    if args.cv_type != 'random':
        visualize_cv_polygons(scv, coords_train_val, meta,
                              f'../figs/cv_{args.cv_type}_{args.n_fold}fold_seed{args.random_state}.tiff')

    # TODO: add accuracy of training set, and add score of test set
    # ### models
    # ## SVC
    # svc = ModelCropland(logger, log_time, 'svc', '1106-101840_svc')
    # # choose from
    # grid search
    # if args.cv_type != 'random':
    #     cv = scv.split(coords_train_val)
    # svc.find_best_parameters(x_train_val, y_train_val, search_by=args.hp_search_by, cv=cv, testing=testing)
    # svc.fit_and_save_best_model(x_train_val, y_train_val)
    # fit known best parameters
    # svc.fit_and_save_best_model(x_train_val, y_train_val, {'C': 100, 'kernel': 'rbf'})
    # # predict and evaluate
    # svc.evaluate_by_metrics(x_test, y_test)
    # svc.predict_and_save(x, meta)
    # svc.evaluate_by_feature_importance(x_test, y_test, feature_names)

    # ## RFC
    rfc = ModelCropland(logger, log_time, 'rfc')
    # # choose from
    # grid search
    if args.cv_type != 'random':
        cv = scv.split(coords_train_val)
    rfc.find_best_parameters(x_train_val, y_train_val, search_by=args.hp_search_by, cv=cv, testing=testing)
    rfc.fit_and_save_best_model(x_train_val, y_train_val)
    # rfc.fit_and_save_best_model(x_train_val, y_train_val,
    #                             {'criterion': 'entropy', 'max_depth': 15, 'max_samples': 0.8, 'n_estimators': 500})
    # predict and evaluate
    rfc.evaluate_by_metrics(x_test, y_test)
    rfc.predict_and_save(x, meta)
    # rfc.evaluate_by_feature_importance(x_test, y_test, feature_names)

    # ### MLP
    mlp = ModelCropland(logger, log_time, 'mlp')
    # # # choose from
    # # grid search
    if args.cv_type != 'random':
        cv = scv.split(coords_train_val)
    mlp.find_best_parameters(x_train_val, y_train_val, search_by=args.hp_search_by, cv=cv, testing=testing)
    mlp.fit_and_save_best_model(x_train_val, y_train_val)
    # # fit known best parameters
    # mlp.fit_and_save_best_model(x_train_val, y_train_val,
    #                             {'hidden_layer_sizes': (100,), 'alpha': 0.0001, 'max_iter': 200,
    #                              'activation': 'relu', 'early_stopping': True})
    # # reload pretrained model
    # # rfc.load_pretrained_model('../models/1008-183014_rfc.sav')
    # # # predict and evaluate
    mlp.evaluate_by_metrics(x_test, y_test)
    mlp.predict_and_save(x, meta)
    mlp.evaluate_by_feature_importance(x_test, y_test, feature_names)

    # ### GRU


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str,
                        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/',
                        help='Base directory to all the images.')
    # cross validation
    parser.add_argument('--cv_type', type=str, default='block', choices=['random', 'block', 'spatial'],
                        help='Method of cross validation.')
    parser.add_argument('--tiles_x', type=int, default=5)
    parser.add_argument('--tiles_y', type=int, default=5)
    parser.add_argument('--shape', type=str, default='square')
    parser.add_argument('--buffer_radius', type=int, default=0)  # TODO: buffer changes to meter
    parser.add_argument('--n_fold', type=int, default=3)
    parser.add_argument('--random_state', type=int, default=42)

    parser.add_argument('--vis_ts', type=bool, default=False)
    parser.add_argument('--vis_profile', type=bool, default=False)
    parser.add_argument('--feature_engineering', type=bool, default=True)
    parser.add_argument('--feature_scaling', type=str, default=None, choices=[None, 'standardize', 'normalize'])
    # hyper parameter
    parser.add_argument('--hp_search_by', type=str, default='grid', choices=['random', 'grid'],
                        help='Method to find hyper-parameters.')

    args = parser.parse_args()

    cropland_classification(args)
