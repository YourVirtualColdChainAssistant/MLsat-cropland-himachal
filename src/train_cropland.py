import argparse
import datetime
from util import *
from util import get_log_dir, get_logger, merge_shapefiles, get_grid_idx
from visualization import NVDI_profile, visualize_train_test_grid_split
from prepare_data import Pipeline, train_test_split, get_spatial_cv_fold
from models import ModelCropland


def classifier_cropland(args):
    # logger
    log_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
    logger = get_logger(get_log_dir(), __name__,
                        f'{log_time}_cropland.log', level='INFO')
    logger.info(args)
    logger.info('----- Cropland Classification -----')
    from_dir = args.images_dir + 'clip/'

    # merge all labels
    merge_shapefiles()
    logger.info('Merged all labels')

    # check NDVI profile
    # ndvi_profile = NVDI_profile(logger, from_dir, '../data/all-labels/all-labels.shp')
    # ndvi_profile.weekly_profile()
    # ndvi_profile.monthly_profile()

    # follow pipeline
    pipe = Pipeline(logger, from_dir)
    df = pipe.pipeline()
    meta = pipe.meta
    num_feature = df.columns.shape[0] - 1
    x = df.iloc[:, :num_feature].values
    df['grid_idx'] = list(get_grid_idx(args.grid_size, meta['height'], meta['width']).reshape(-1))

    # train val test split
    if args.dataset_split_by == 'spatial':
        logger.info('Spatial train-val-test split...')
        x_train_val, x_test, y_train_val, y_test, grid_idx_train_val, grid_idx_test = \
            train_test_split(logger, df, df['label'] != 0, num_feature, split_by='spatial',
                             test_ratio=0.2, random_seed=args.random_seed)
        data_cv, grid_idx_fold = get_spatial_cv_fold(x_train_val, y_train_val, grid_idx_train_val)
        visualize_train_test_grid_split(meta, args.grid_size, grid_idx_test, [grid_idx_train_val],
                                        f'../figs/cropland_{args.grid_size}_train_test_split_{args.random_seed}.tiff')
        logger.info('Saved train-test visualization.')
        visualize_train_test_grid_split(meta, args.grid_size, grid_idx_test, grid_idx_fold,
                                        f'../figs/cropland_{args.grid_size}_train_val_test_split_{args.random_seed}.tiff')
        logger.info('Saved train-val-test visualization.')
    else:
        logger.info('Random train-val-test split...')
        x_train_val, x_test, y_train_val, y_test, grid_idx_train_val, grid_idx_test = \
            train_test_split(logger, df, df['label'] != 0, num_feature, split_by='random',
                             test_ratio=0.2)

    # print
    feature_names = df.columns[:num_feature]
    logger.info(f'\nFeatures: {feature_names}')
    logger.info(f'  x_train_val.shape {x_train_val.shape}, y_train_val.shape {y_train_val.shape}')
    logger.info(f'  x_test.shape {x_test.shape}, y_test.shape {y_test.shape}')

    # ### models
    # ## SVC
    svc = ModelCropland(logger, log_time, 'svc')
    # # choose from
    # grid search
    if args.dataset_split_by == 'spatial':
        svc.find_best_parameters(x_train_val, y_train_val, search_by=args.cv_search_by, cv=data_cv)
    else:
        svc.find_best_parameters(x_train_val, y_train_val, search_by=args.cv_search_by)
    svc.fit_and_save_best_model(x_train_val, y_train_val)
    # fit known best parameters
    # svc.fit_and_save_best_model(x_train_val, y_train_val, {'C': 100, 'kernel': 'rbf'})
    # # predict and evaluate
    svc.predict_and_save(x, meta)
    svc.evaluate(x_test, y_test, feature_names)

    # ## RFC
    rfc = ModelCropland(logger, log_time, 'rfc')
    # # choose from
    # grid search
    if args.dataset_split_by == 'random':
        rfc.find_best_parameters(x_train_val, y_train_val, search_by=args.cv_search_by)
    elif args.dataset_split_by == 'spatial':
        rfc.find_best_parameters(x_train_val, y_train_val, search_by=args.cv_search_by, cv=data_cv)
    rfc.fit_and_save_best_model(x_train_val, y_train_val)
    # rfc.fit_and_save_best_model(x_train_val, y_train_val,
    #                             {'criterion': 'entropy', 'max_depth': 15, 'max_samples': 0.8, 'n_estimators': 500})
    # predict and evaluate
    rfc.save_predictions(x, meta)
    rfc.evaluate(x_test, y_test, feature_names)

    # ### MLP
    mlp = ModelCropland(logger, log_time, 'mlp')
    # # # choose from
    # # grid search
    if args.dataset_split_by == 'random':
        mlp.find_best_parameters(x_train_val, y_train_val, search_by=args.cv_search_by)
    elif args.dataset_split_by == 'spatial':
        mlp.find_best_parameters(x_train_val, y_train_val, search_by=args.cv_search_by, cv=data_cv)
    mlp.fit_and_save_best_model(x_train_val, y_train_val)
    # # fit known best parameters
    # mlp.fit_and_save_best_model(x_train_val, y_train_val,
    #                             {'hidden_layer_sizes': (100,), 'alpha': 0.0001, 'max_iter': 200,
    #                              'activation': 'relu', 'early_stopping': True})
    # # reload pretrained model
    # # rfc.load_pretrained_model('../models/1008-183014_rfc.sav')
    # # # predict and evaluate
    mlp.save_predictions(x, meta)
    mlp.evaluate(x_test, y_test, feature_names)

    # ### GRU


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str,
                        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/',
                        help='Base directory to all the images.')
    parser.add_argument('--dataset_split_by', type=str, default='spatial', choices=['random', 'spatial'],
                        help='Method to split train-val-test dataset.')
    parser.add_argument('--cv_search_by', type=str, default='grid', choices=['random', 'grid'],
                        help='Method to do cross validation.')
    parser.add_argument('--grid_size', type=int, default=1000,
                        help='Size of grid during spatial split.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random see for train_test_split.')
    args = parser.parse_args()
    classifier_cropland(args)
