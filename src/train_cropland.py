import argparse
import datetime
# import skgstat as skg
import numpy as np
import geopandas as gpd
from src.data.prepare_data import prepare_data, construct_grid_to_fold, clean_train_shapefiles
from src.models.cropland import CroplandModel
from src.utils.logger import get_log_dir, get_logger
from src.utils.scv import ModifiedBlockCV, ModifiedSKCV
from src.evaluation.visualize import visualize_cv_fold, visualize_cv_polygons


def cropland_classification(args):
    testing = False

    # logger
    log_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
    log_filename = f'cropland_{log_time}.log' if not testing else f'cropland_testing_{log_time}.log'
    logger = get_logger(get_log_dir(), __name__, log_filename, level='INFO')
    logger.info(args)

    logger.info('#### Cropland Classification')
    clean_train_shapefiles()
    feature_dir = args.img_dir + args.tile_id + '/raster/' if not testing else args.img_dir + args.tile_id + '/raster_sample/'

    # prepare train and validation dataset
    df_tv, df_train_val, x_train_val, y_train_val, polygons, _, scaler, meta, n_feature, feature_names = \
        prepare_data(logger=logger, dataset='train_val', feature_dir=feature_dir, task='cropland', window=None,
                     label_path='../data/train_labels/train_labels.shp',
                     feature_engineering=args.feature_engineering, scaling=args.scaling, smooth=args.smooth,
                     fill_missing=args.fill_missing, check_missing=args.check_missing,
                     vis_stack=args.vis_stack, vis_profile=args.vis_profile)
    coords_train_val = gpd.GeoDataFrame({'geometry': df_train_val.coords.values})
    x = df_tv.iloc[:, :n_feature].values
    if args.scaling == 'standardize' or args.scaling == 'normalize':
        x = scaler.transform(x)
        logger.info('Transformed all x')

    # if args.check_autocorrelation:
    #     # TODO: draw more pairs below 5km, see the values of auto-correlation
    #     # TODO: how many pixels are we moving if use buffer_radius=3km as suggested in the semi-variogram
    #     # visualize autocorrelation
    #     sample_ids_equal = np.linspace(0, df_train_val.shape[0], num=2000, dtype=int, endpoint=False)
    #     sample_ids_near = np.arange(2000)
    #     # equal
    #     coords_xy = np.array([[coord.x, coord.y] for coord in df_train_val.coords.values])
    #     variogram_equal = skg.Variogram(coords_xy[sample_ids_equal], df_train_val.gt_cropland.values[sample_ids_equal])
    #     variogram_equal.plot().savefig('../figs/semivariogram_equal.png', bbox_inches='tight')
    #     # near
    #     variogram_near = skg.Variogram(coords_xy[sample_ids_near], df_train_val.gt_cropland.values[sample_ids_near])
    #     variogram_near.plot().savefig('../figs/semivariogram_near.png', bbox_inches='tight')
    #     logger.info('Saved semivariogram')

    # TODO: how to use semi-variogram or other statistics in practice
    # TODO: why to study the effects of buffer radius
    # set cross validation
    if args.cv_type == 'random':
        cv = args.n_fold
    elif args.cv_type == 'block':
        # assign to fold
        grid = construct_grid_to_fold(polygons, tiles_x=args.tiles_x, tiles_y=args.tiles_y, shape=args.shape,
                                      data=x_train_val, n_fold=args.n_fold, random_state=args.random_state)
        scv = ModifiedBlockCV(custom_polygons=grid, buffer_radius=args.buffer_radius)
        # visualize valid block
        cv_name = f'../figs/cv_{args.tiles_x}x{args.tiles_y}{args.shape}_f{args.n_fold}_s{args.random_state}'
        visualize_cv_fold(grid, meta, cv_name + '.tiff')
        logger.info(f'Saved {cv_name}.tiff')
        visualize_cv_polygons(scv, coords_train_val, meta, cv_name + '_mask.tiff')
        logger.info(f'Saved {cv_name}_mask.tiff')
    elif args.cv_type == 'spatial':
        scv = ModifiedSKCV(n_splits=args.n_fold, buffer_radius=args.buffer_radius, random_state=args.random_state)
        cv_name = f'../figs/cv_{args.cv_type}_f{args.n_fold}_s{args.random_state}'
        visualize_cv_polygons(scv, coords_train_val, meta, cv_name + '_mask.tiff')
        logger.info(f'Saved {cv_name}_mask.tiff')

    # models
    best_params = {
        'svc': {'C': 0.5, 'gamma': 'scale', 'kernel': 'poly', 'random_state': args.random_state},
        'rfc': {'criterion': 'entropy', 'max_depth': 15, 'max_samples': 0.8, 'n_estimators': 500,
                'random_state': args.random_state},
        'mlp': {'hidden_layer_sizes': (100,), 'alpha': 0.0001, 'max_iter': 200, 'activation': 'relu',
                'early_stopping': True, 'random_state': args.random_state}
    }
    for m in best_params.keys():
        model = CroplandModel(logger, log_time, m, args.random_state)
        if args.cv_type is None:
            model.fit_best(x_train_val, y_train_val, best_params[m])
        elif args.cv_type == 'random':
            model.find_best_hyperparams(x_train_val, y_train_val, search_by=args.hp_search_by, cv=cv, testing=testing)
            model.fit_best(x_train_val, y_train_val)
        else:  # cv_type == 'block' or 'spatial'
            cv = scv.split(coords_train_val)
            model.find_best_hyperparams(x_train_val, y_train_val, search_by=args.hp_search_by, cv=cv, testing=testing)
            model.fit_best(x_train_val, y_train_val)
        # predict and evaluation
        model.test(x_train_val, y_train_val, meta, index=df_train_val.index,
                   region_shp_path='../data/train_labels/train_labels.shp',
                   feature_names=None, pred_name=f'{log_time}_{m}_train')
        model.predict(x, meta, region_shp_path='../data/train_region/train_region.shp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: add a configuration file and save with the same file name
    parser.add_argument('--img_dir', type=str,
                        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/',
                        help='Base directory to all the images.')
    parser.add_argument('--tile_id', type=str, default='43SFR')

    # cross validation
    parser.add_argument('--cv_type', type=str, default=None, choices=[None, 'random', 'block', 'spatial'],
                        help='Method of cross validation.')
    parser.add_argument('--tiles_x', type=int, default=4)
    parser.add_argument('--tiles_y', type=int, default=4)
    parser.add_argument('--shape', type=str, default='square')
    parser.add_argument('--buffer_radius', type=int, default=0)  # TODO: buffer changes to meter
    parser.add_argument('--n_fold', type=int, default=3)
    parser.add_argument('--random_state', type=int, default=24)

    # hyper parameter
    parser.add_argument('--hp_search_by', type=str, default='grid', choices=['random', 'grid'],
                        help='Method to find hyper-parameters.')

    # prepare data
    parser.add_argument('--vis_stack', type=bool, default=False)
    parser.add_argument('--vis_profile', type=bool, default=True)
    parser.add_argument('--feature_engineering', type=bool, default=True)
    parser.add_argument('--smooth', type=bool, default=False)
    parser.add_argument('--scaling', type=str, default='as_float',
                        choices=['as_float', 'as_TOA', 'standardize', 'normalize'])
    parser.add_argument('--fill_missing', type=str, default='linear', choices=[None, 'forward', 'linear'])
    parser.add_argument('--check_missing', type=bool, default=False)
    parser.add_argument('--check_autocorrelation', type=bool, default=False)

    args = parser.parse_args()

    cropland_classification(args)
