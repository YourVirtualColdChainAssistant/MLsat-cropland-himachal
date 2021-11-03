import argparse
import datetime
from spacv.spacv import SKCV

from util import get_log_dir, get_logger
from visualization import NVDI_profile, visualize_valid_block, visualize_cv
from prepare_data import clean_train_shapefiles, clean_test_shapefiles, prepare_data
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

    # check NDVI profile
    # ndvi_profile = NVDI_profile(logger, train_val_dir, '../data/labels/labels.shp')
    # ndvi_profile.weekly_profile()
    # ndvi_profile.monthly_profile()

    # prepare train/validation/test set
    df_tv, df_train_val, x_train_val, y_train_val, coords_train_val, scaler, meta, n_feature, feature_names = \
        prepare_data(logger, dataset='train_val', feature_dir=train_val_dir,
                     label_path='../data/train_labels/train_labels.shp',
                     feature_scaling=args.feature_scaling)
    df_te, df_test, x_test, y_test, _, _, _, _, _ = \
        prepare_data(logger, dataset='test', feature_dir=test_dir,
                     label_path='../data/test_labels/test_labels.shp',
                     feature_scaling=args.feature_scaling, scaler=scaler)
    logger.info(f'\nFeatures: {feature_names}')
    n_train_val = df_train_val.shape[0]
    x = df_tv.iloc[:, :n_feature].values
    if args.feature_scaling is not None:
        x = scaler.transform(x)

    # cross validation
    if args.cv_type == 'random':
        cv = args.n_fold
    elif args.cv_type == 'block':
        scv = ModifiedBlockCV(tiles_x=args.tiles_x, tiles_y=args.tiles_y, shape=args.shape, method='random',
                              buffer_radius=args.buffer_radius, n_groups=args.n_fold, data=x_train_val,
                              random_state=args.random_state)
        cv = scv.split(coords_train_val)
        # visualize valid block
        _, valid_block, _ = scv.construct_valid_block(coords_train_val)
        visualize_valid_block(valid_block, meta,
                              save_path=f'../figs/block_{args.tiles_x}x{args.tiles_y}{args.shape}_{args.n_fold}fold_seed{args.random_state}.tiff')
    else:  # spatial
        scv = SKCV(n_splits=args.n_splits, buffer_radius=args.buffer_radius, random_state=args.random_state)
        cv = scv.split(coords_train_val)

    if args.cv_type != 'random':
        visualize_cv(scv, coords_train_val, n_train_val, meta,
                     f'../figs/{args.cv_type}CV_{args.n_fold}fold_seed{args.random_state}.tiff')

    # ### models
    # ## SVC
    svc = ModelCropland(logger, log_time, 'svc')
    # # choose from
    # grid search
    svc.find_best_parameters(x_train_val, y_train_val, search_by=args.hp_search_by, cv=cv, testing=testing)
    svc.fit_and_save_best_model(x_train_val, y_train_val)
    # fit known best parameters
    # svc.fit_and_save_best_model(x_train_val, y_train_val, {'C': 100, 'kernel': 'rbf'})
    # # predict and evaluate
    svc.evaluate(x_test, y_test, feature_names)
    svc.predict_and_save(x, meta)

    # ## RFC
    rfc = ModelCropland(logger, log_time, 'rfc')
    # # choose from
    # grid search
    rfc.find_best_parameters(x_train_val, y_train_val, search_by=args.hp_search_by, cv=cv, testing=testing)
    rfc.fit_and_save_best_model(x_train_val, y_train_val)
    # rfc.fit_and_save_best_model(x_train_val, y_train_val,
    #                             {'criterion': 'entropy', 'max_depth': 15, 'max_samples': 0.8, 'n_estimators': 500})
    # predict and evaluate
    rfc.evaluate(x_test, y_test, feature_names)
    rfc.predict_and_save(x, meta)

    # ### MLP
    mlp = ModelCropland(logger, log_time, 'mlp')
    # # # choose from
    # # grid search
    mlp.find_best_parameters(x_train_val, y_train_val, search_by=args.hp_search_by, cv=cv, testing=testing)
    mlp.fit_and_save_best_model(x_train_val, y_train_val)
    # # fit known best parameters
    # mlp.fit_and_save_best_model(x_train_val, y_train_val,
    #                             {'hidden_layer_sizes': (100,), 'alpha': 0.0001, 'max_iter': 200,
    #                              'activation': 'relu', 'early_stopping': True})
    # # reload pretrained model
    # # rfc.load_pretrained_model('../models/1008-183014_rfc.sav')
    # # # predict and evaluate
    mlp.evaluate(x_test, y_test, feature_names)
    mlp.predict_and_save(x, meta)

    # ### GRU


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str,
                        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/',
                        help='Base directory to all the images.')
    # cross validation
    parser.add_argument('--cv_type', type=str, default='block', choices=['random', 'block', 'spatial'],
                        help='Method of cross validation.')
    parser.add_argument('--tiles_x', type=int, default=4)
    parser.add_argument('--tiles_y', type=int, default=4)
    parser.add_argument('--shape', type=str, default='square')
    parser.add_argument('--buffer_radius', type=int, default=0)
    parser.add_argument('--n_fold', type=int, default=3)
    parser.add_argument('--random_state', type=int, default=42)

    parser.add_argument('--feature_engineering', type=bool, default=True)
    parser.add_argument('--feature_scaling', type=str, default=None, choices=[None, 'standardize', 'normalize'])
    # hyper parameter
    parser.add_argument('--hp_search_by', type=str, default='grid', choices=['random', 'grid'],
                        help='Method to find hyper-parameters.')

    args = parser.parse_args()

    cropland_classification(args)
