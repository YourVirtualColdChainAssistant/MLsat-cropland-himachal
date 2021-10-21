import argparse
import datetime
from util import *
from util import get_log_dir, get_logger, merge_shapefiles
from visualization import NVDI_profile, visualize_train_test_grid_split
from prepare_data import Pipeline, train_test_split, spatial_cross_validation
from models import Model


def train(args):
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
    x = df.iloc[:, :-1].values

    # train val test split
    if args.dataset_split == 'random':
        logger.info('Random train-val-test split...')
        x_train_val, x_test, y_train_val, y_test, grid_idx_train_val, grid_idx_test = \
            train_test_split(logger, df, by='random', spatial_dict=None, test_ratio=0.2)
    elif args.dataset_split == 'spatial':
        logger.info('Spatial train-val-test split...')
        spatial_dict = {'cell_size': 64, 'height': meta['height'], 'width': meta['width']}
        x_train_val, x_test, y_train_val, y_test, grid_idx_train_val, grid_idx_test = \
            train_test_split(logger, df, by='spatial', spatial_dict=spatial_dict, test_ratio=0.2)
        data_cv, grid_idx_fold = spatial_cross_validation(x_train_val, y_train_val, grid_idx_train_val)
        # visualize_train_test_grid_split(meta, spatial_dict, grid_idx_test, [grid_idx_train_val],
        #                                 '../preds/train_test_split.tiff')
        # logger.info('Saved train-test visualization.')
        visualize_train_test_grid_split(meta, spatial_dict, grid_idx_test, grid_idx_fold,
                                        '../preds/train_val_test_split.tiff')
        logger.info('Saved train-val-test visualization.')

    # print
    feature_names = df.columns[:-1]
    logger.info(f'\nFeatures: {feature_names}')
    logger.info(f'  x_train_val.shape {x_train_val.shape}, y_train_val.shape {y_train_val.shape}')
    logger.info(f'  x_test.shape {x_test.shape}, y_test.shape {y_test.shape}')

    # ### models
    # ## SVC
    svc = Model(logger, log_time, 'svc')
    # # choose from
    # grid search
    # if args.dataset_split == 'random':
    #     svc.find_best_parameters(x_train_val, y_train_val, by='grid')
    # else:
    #     svc.find_best_parameters(x_train_val, y_train_val, by='grid', cv=data_cv)
    # svc.fit_and_save_best_model(x_train_val, y_train_val)
    # fit known best parameters
    svc.fit_and_save_best_model(x_train_val, y_train_val, {'C': 100, 'kernel': 'rbf'})
    # reload pretrained model
    # svc.load_pretrained_model('../models/1015-172420_svc.sav')
    # # predict and evaluate
    svc.save_predictions(x, meta)
    svc.evaluate_all(x_test, y_test, feature_names)

    # ## RFC
    rfc = Model(logger, log_time, 'rfc')
    # # choose from
    # grid search
    # if args.dataset_split == 'random':
    #     rfc.find_best_parameters(x_train_val, y_train_val, by='random')
    # elif args.dataset_split == 'spatial':
    #     rfc.find_best_parameters(x_train_val, y_train_val, by='random', cv=data_cv)
    # fit known best parameters
    rfc.fit_and_save_best_model(x_train_val, y_train_val,
                                {'criterion': 'entropy', 'max_depth': 15, 'max_samples': 0.8, 'n_estimators': 500})
    # reload pretrained model
    # rfc.load_pretrained_model('../models/1008-183014_rfc.sav')
    # # predict and evaluate
    rfc.save_predictions(x, meta)
    rfc.evaluate_all(x_test, y_test, feature_names)

    # ### MLP
    # mlp = Model(logger, log_time, 'mlp')
    # # # choose from
    # # grid search
    # mlp.find_best_parameters(x_train_val, y_train_val, 'random')
    # mlp.fit_and_save_best_model(x_train_val, y_train_val)
    # # fit known best parameters
    # # mlp.fit_and_save_best_model(x_train_val, y_train_val,
    # #   todo: an argument )
    # # reload pretrained model
    # # rfc.load_pretrained_model('../models/1008-183014_rfc.sav')
    # # # predict and evaluate
    # mlp.save_predictions(x, meta)
    # mlp.evaluate_all(x_test, y_test, feature_names)

    # ### GRU


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images_dir',
        type=str,
        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/',
        help='Base directory to all the images.'
    )
    parser.add_argument(
        '--dataset_split',
        type=str,
        default='spatial',
        choices=['random', 'spatial'],
        help='The way to split train-val-test dataset.'
    )
    args = parser.parse_args()
    train(args)
