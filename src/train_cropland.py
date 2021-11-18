import argparse
import datetime
import skgstat as skg
import numpy as np
import geopandas as gpd
# from pysal.explore import esda
# from libpysal.weights import W
from src.data.prepare_data import prepare_data, construct_grid_to_fold, clean_train_shapefiles, clean_test_shapefiles
from src.models.cropland import CroplandModel
from src.utils.logger import get_log_dir, get_logger
from src.utils.scv import ModifiedBlockCV, ModifiedSKCV
from src.evaluation.visualize import visualize_cv_fold, visualize_cv_polygons


def cropland_classification(args):
    testing = True
    tile_dir = args.img_dir + args.tile_id + '/'
    # logger
    log_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
    log_filename = f'cropland_{log_time}.log' if not testing else f'cropland_testing_{log_time}.log'
    logger = get_logger(get_log_dir(), __name__, log_filename, level='INFO')
    logger.info(args)

    logger.info('#### Cropland Classification')
    train_val_dir = tile_dir + 'train_region/' if not testing else tile_dir + 'train_region_sample/'
    test_dir = tile_dir + 'test_region_near/' if not testing else tile_dir + 'test_region_near_sample/'

    # clean shapefiles
    logger.info('# Clean shapefiles')
    clean_train_shapefiles()
    clean_test_shapefiles()
    logger.info('  ok')

    # prepare train/validation/test set
    df_tv, df_train_val, x_train_val, y_train_val, polygons, scaler, meta, n_feature, feature_names = \
        prepare_data(logger, dataset='train_val', feature_dir=train_val_dir,
                     label_path='../data/train_labels/train_labels.shp',
                     feature_engineering=args.feature_engineering,
                     scaling=args.scaling,
                     vis_ts=args.vis_ts, vis_profile=args.vis_profile)
    df_te, df_test, x_test, y_test, _, _, _, _, _ = \
        prepare_data(logger, dataset='test', feature_dir=test_dir,
                     label_path='../data/test_labels/test_labels.shp',
                     feature_engineering=args.feature_engineering,
                     scaling=args.scaling, scaler=scaler,
                     vis_ts=args.vis_ts, vis_profile=args.vis_profile)
    logger.info(f'\nFeatures: {feature_names}')
    coords_train_val = gpd.GeoDataFrame({'geometry': df_train_val.coords.values})
    x = df_tv.iloc[:, :n_feature].values
    if args.scaling is not None:
        x = scaler.transform(x)
        logger.info('Transformed all x')

    # plot semivariogram
    sample_ids_equal = np.linspace(0, df_train_val.shape[0], num=2000, dtype=int, endpoint=False)
    sample_ids_near = np.arange(2000)
    # equal
    coords_xy = np.array([[coord.x, coord.y] for coord in df_train_val.coords.values])
    variogram_equal = skg.Variogram(coords_xy[sample_ids_equal], df_train_val.gt_cropland.values[sample_ids_equal])
    variogram_equal.plot().savefig('../figs/semivariogram_equal.png', bbox_inches='tight')
    # near
    variogram_near = skg.Variogram(coords_xy[sample_ids_near], df_train_val.gt_cropland.values[sample_ids_near])
    variogram_near.plot().savefig('../figs/semivariogram_near.png', bbox_inches='tight')
    logger.info('Saved semivariogram')

    # calculate Moran's I
    # equal
    # neighbors_equal = df_train_val.neighbors[sample_ids_equal].to_dict()
    # weights_equal = df_train_val.neighbors[sample_ids_equal].apply(lambda x: [1 / len(x) for _ in x])
    # w_equal = W(neighbors_equal, weights_equal)
    # moran_equal = esda.Moran(df_train_val.gt_cropland.values[sample_ids_equal], w_equal)
    # logger.info(f'*equal: \n  moran.I {moran_equal.I}, moran.EI {moran_equal.EI}, moran.p_sim {moran_equal.p_sim},' + \
    #             f'  moran.EI_sim {moran_equal.EI_sim}, moran.z_sim {moran_equal.z_sim}, moran.p_z_sim {moran_equal.p_z_sim}')
    # # near
    # neighbors_near = df_train_val.neighbors[sample_ids_near].to_dict()
    # weights_near = df_train_val.neighbors[sample_ids_near].apply(lambda x: [1 / len(x) for _ in x])
    # w_near = W(neighbors_near, weights_near)
    # moran_near = esda.Moran(df_train_val.gt_cropland.values[sample_ids_near], w_near)
    # logger.info(f'*near: \n  moran.I {moran_near.I}, moran.EI {moran_near.EI}, moran.p_sim {moran_near.p_sim},' + \
    #             f'  moran.EI_sim {moran_near.EI_sim}, moran.z_sim {moran_near.z_sim}, moran.p_z_sim {moran_near.p_z_sim}')

    # cross validation
    if args.cv_type == 'random':
        cv = args.n_fold
    elif args.cv_type == 'block':
        # assign to fold
        grid = construct_grid_to_fold(polygons, tiles_x=args.tiles_x, tiles_y=args.tiles_y, shape=args.shape,
                                      data=x_train_val, n_fold=args.n_fold, random_state=args.random_state)
        scv = ModifiedBlockCV(custom_polygons=grid, buffer_radius=args.buffer_radius)
        # visualize valid block
        cv_name = f'../figs/cv_{args.tiles_x}x{args.tiles_y}{args.shape}_f{args.n_fold}_s{args.random_state}'
        logger.info(f'Saving cv block to {cv_name}.tiff')
        visualize_cv_fold(grid, meta, cv_name + '.tiff')
        logger.info(f' ok')
    else:  # spatial
        scv = ModifiedSKCV(n_splits=args.n_fold, buffer_radius=args.buffer_radius, random_state=args.random_state)
        cv_name = f'../figs/cv_{args.cv_type}_f{args.n_fold}_s{args.random_state}'

    if args.cv_type != 'random':
        logger.info(f'Saving cv polygons to {cv_name}_mask.tiff')
    visualize_cv_polygons(scv, coords_train_val, meta, cv_name + '_mask.tiff')
    logger.info(f' ok')

    # ### models
    # ## SVC
    svc = CroplandModel(logger, log_time, 'svc', args.random_state)
    # # choose from
    # grid search
    # if args.cv_type != 'random':
    #     cv = scv.split(coords_train_val)
    # svc.find_best_parameters(x_train_val, y_train_val, search_by=args.hp_search_by, cv=cv, testing=testing)
    # svc.fit_and_save_best_model(x_train_val, y_train_val)
    # fit known best parameters
    svc.fit_and_save_best_model(x_train_val, y_train_val,
                                {'C': 0.5, 'gamma': 'scale', 'kernel': 'poly', 'random_state': args.random_state})
    # predict and evaluation
    svc.evaluate_by_metrics(x_test, y_test)
    svc.predict_and_save(x, meta)
    svc.evaluate_by_feature_importance(x_test, y_test, feature_names)

    # ## RFC
    rfc = CroplandModel(logger, log_time, 'rfc', args.random_state)
    # # # choose from
    # # grid search
    # if args.cv_type != 'random':
    #     cv = scv.split(coords_train_val)
    # rfc.find_best_parameters(x_train_val, y_train_val, search_by=args.hp_search_by, cv=cv, testing=testing)
    # rfc.fit_and_save_best_model(x_train_val, y_train_val)
    rfc.fit_and_save_best_model(x_train_val, y_train_val,
                                {'criterion': 'entropy', 'max_depth': 15, 'max_samples': 0.8, 'n_estimators': 500,
                                 'random_state': args.random_state})
    # predict and evaluation
    rfc.evaluate_by_metrics(x_test, y_test)
    rfc.predict_and_save(x, meta)
    rfc.evaluate_by_feature_importance(x_test, y_test, feature_names)

    # ### MLP
    mlp = CroplandModel(logger, log_time, 'mlp', args.random_state)
    # # # choose from
    # # grid search
    # if args.cv_type != 'random':
    #     cv = scv.split(coords_train_val)
    # mlp.find_best_parameters(x_train_val, y_train_val, search_by=args.hp_search_by, cv=cv, testing=testing)
    # mlp.fit_and_save_best_model(x_train_val, y_train_val)
    # # fit known best parameters
    mlp.fit_and_save_best_model(x_train_val, y_train_val,
                                {'hidden_layer_sizes': (100,), 'alpha': 0.0001, 'max_iter': 200,
                                 'activation': 'relu', 'early_stopping': True, 'random_state': args.random_state})
    # # reload pretrained model
    # # rfc.load_pretrained_model('../models/1008-183014_rfc.sav')
    # # # predict and evaluation
    # # # predict and evaluation
    mlp.evaluate_by_metrics(x_test, y_test)
    mlp.predict_and_save(x, meta)
    mlp.evaluate_by_feature_importance(x_test, y_test, feature_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str,
                        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/',
                        help='Base directory to all the images.')
    parser.add_argument('--tile_id', type=str, default='43SFR')

    # cross validation
    parser.add_argument('--cv_type', type=str, default='block', choices=['random', 'block', 'spatial'],
                        help='Method of cross validation.')
    parser.add_argument('--tiles_x', type=int, default=4)
    parser.add_argument('--tiles_y', type=int, default=4)
    parser.add_argument('--shape', type=str, default='square')
    parser.add_argument('--buffer_radius', type=int, default=0)  # TODO: buffer changes to meter
    parser.add_argument('--n_fold', type=int, default=3)
    parser.add_argument('--random_state', type=int, default=24)

    parser.add_argument('--vis_ts', type=bool, default=True)
    parser.add_argument('--vis_profile', type=bool, default=True)
    parser.add_argument('--feature_engineering', type=bool, default=True)
    parser.add_argument('--scaling', type=str, default='standardize', choices=[None, 'standardize', 'normalize'])
    # hyper parameter
    parser.add_argument('--hp_search_by', type=str, default='grid', choices=['random', 'grid'],
                        help='Method to find hyper-parameters.')

    args = parser.parse_args()

    cropland_classification(args)
