import os
import pickle
import argparse
import datetime
import numpy as np
import glob
from rasterio.windows import Window
from rasterio.merge import merge
from src.data.prepare_data import prepare_data, clean_test_shapefiles
from src.utils.logger import get_log_dir, get_logger
from src.utils.util import save_predictions_geotiff
from src.models.cropland import CroplandModel


def cropland_predict(args):
    testing = False

    # logger
    log_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
    log_filename = f'cropland_{log_time}_predict.log' if not testing else f'cropland_testing_{log_time}_predict.log'
    logger = get_logger(get_log_dir(), __name__, log_filename, level='INFO')
    logger.info(args)

    logger.info('#### Test Cropland Model')
    train_val_dir = args.img_dir + '43SFR/raster/' if not testing else args.img_dir + '43SFR/raster_sample/'
    test_near_dir = args.img_dir + '43SFR/raster/' if not testing else args.img_dir + '43SFR/raster_sample/'
    test_far_dir = args.img_dir + '43RGQ/raster/' if not testing else args.img_dir + '43RGQ/raster_sample/'
    predict_dir = args.img_dir + args.tile_id + '/raster/' if not testing else args.img_dir + args.tile_id + '/raster_sample/'

    # _, _, _, _, _, scaler, _, _, _ = \
    #     prepare_data(logger=logger, dataset='train_val', feature_dir=train_val_dir, task='cropland', window=None,
    #                  label_path='../data/train_labels/train_labels.shp',
    #                  feature_engineering=args.feature_engineering, scaling=args.scaling, smooth=args.smooth,
    #                  fill_missing=args.fill_missing, check_missing=False,
    #                  vis_stack=False, vis_profile=False)
    scaler = None

    if not args.test_far:
        # load pretrained model
        logger.info("Loading the best pretrained model...")
        model = pickle.load(open(f'../models/{args.pretrained}.pkl', 'rb'))

        # read and predict in patches
        full_len, n_patch = 10800, 10
        patch_len = int(full_len / n_patch)

        # check path existence
        file_path = f'../preds/tiles/{args.tile_id}/'
        if not os.path.isdir(file_path):
            os.makedirs(file_path)

        # predict patch by patch
        for row in np.linspace(0, full_len, n_patch, endpoint=False, dtype=int):
            for col in np.linspace(0, full_len, n_patch, endpoint=False, dtype=int):
                # get window
                window = Window(col, row, patch_len, patch_len)  # (col_off, row_off, width, height)
                logger.info(f'==== Preparing for {window} ====')

                # prepare data
                df, x, meta, n_feature, feature_names = \
                    prepare_data(logger=logger, dataset='predict', feature_dir=predict_dir,
                                 label_path=None, window=window, task='cropland', smooth=args.smooth,
                                 feature_engineering=args.feature_engineering, scaling=args.scaling, scaler=scaler,
                                 fill_missing=args.fill_missing, check_missing=True,
                                 vis_stack=args.vis_stack, vis_profile=args.vis_profile)
                logger.info(f'df.shape {df.shape}, x.shape {x.shape}')

                # predict
                logger.info('Predicting...')
                preds = model.predict(x)
                # save predictions
                logger.info('Saving predictions...')
                pred_name = file_path + str(row) + '_' + str(col) + '.tiff'
                save_predictions_geotiff(meta, preds, pred_name)
                logger.info(f'Predictions are saved to {pred_name}')

        # merge patches into single raster
        logger.info('Merging patches...')
        patches_list = [f for f in glob.glob(file_path + '*.tiff')]
        merge(patches_list, dst_path=file_path + args.tile_id + '.tiff')
    else:
        test_dir_dict = {'kullu': test_near_dir, 'mandi': test_far_dir, 'shimla': test_far_dir}
        # test_dir_dict = {'mandi': test_far_dir, 'shimla': test_far_dir}
        clean_test_shapefiles()
        for district in test_dir_dict.keys():
            logger.info(f'### Test on {district}')
            test_dir = test_dir_dict[district]
            label_path = f'../data/test_labels_{district}/test_labels_{district}.shp'
            # prepare data
            _, df_test, x_test, y_test, _, _, _, meta, n_feature, feature_names = \
                prepare_data(logger=logger, dataset=f'test_{district}', feature_dir=test_dir,
                             label_path=label_path, window=None, task='cropland',
                             feature_engineering=args.feature_engineering, scaling=args.scaling, scaler=scaler,
                             fill_missing=args.fill_missing, check_missing=True, smooth=args.smooth,
                             vis_stack=args.vis_stack, vis_profile=args.vis_profile)
            # test
            model = CroplandModel(logger, log_time, args.pretrained.split('_')[-1],
                                  args.random_state, pretrained_name=args.pretrained)
            model.test(x_test, y_test, meta, index=df_test.index, region_shp_path=label_path,
                       feature_names=feature_names, pred_name=f'{args.pretrained}_{district}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str,
                        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/',
                        help='Base directory to all the images.')
    parser.add_argument('--tile_id', type=str, default='43RGQ')
    parser.add_argument('--test_far', type=bool, default=True)
    parser.add_argument('--pretrained', type=str, default='1119-224829_svc',
                        help='Filename of the best pretrained models.')
    parser.add_argument('--random_state', type=int, default=24)

    parser.add_argument('--vis_stack', type=bool, default=True)
    parser.add_argument('--vis_profile', type=bool, default=True)
    parser.add_argument('--feature_engineering', type=bool, default=True)
    parser.add_argument('--smooth', type=bool, default=False)
    parser.add_argument('--scaling', type=str, default=None, choices=[None, 'to_TOA', 'standardize', 'normalize'])
    parser.add_argument('--fill_missing', type=str, default='forward', choices=[None, 'forward', 'linear'])

    args = parser.parse_args()

    cropland_predict(args)
