import pickle
import argparse
import datetime
from src.data.prepare_data import prepare_data
from src.utils.util import get_log_dir, get_logger, save_predictions_geotiff


def cropland_predict(args):
    testing = True
    tile_dir = args.img_dir + args.tile_id + '/'
    # logger
    log_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
    log_filename = f'cropland_{log_time}_predict.log' if not testing else f'cropland_testing_{log_time}_predict.log'
    logger = get_logger(get_log_dir(), __name__, log_filename, level='INFO')
    logger.info(args)

    logger.info('#### Test Cropland Model')
    train_val_dir = args.img_dir + '43SFR/train_area/' if not testing else args.img_dir + '43SFR/train_area_sample/'
    predict_dir = tile_dir + 'geotiff/' if not testing else tile_dir + 'geotiff_sample/'

    # prepare train/validation/test set
    _, _, _, _, _, scaler, _, _, _ = \
        prepare_data(logger, dataset='train_val', feature_dir=train_val_dir,
                     label_path='../data/train_labels/train_labels.shp',
                     feature_engineering=args.feature_engineering,
                     scaling=args.scaling,
                     vis_ts=False, vis_profile=False)
    df, x, meta, n_feature, feature_names = \
        prepare_data(logger, dataset='predict', feature_dir=predict_dir,
                     label_path='../data/test_labels/test_labels.shp',
                     feature_engineering=args.feature_engineering,
                     scaling=args.scaling, scaler=scaler,
                     vis_ts=args.vis_ts, vis_profile=args.vis_profile)
    logger.info(f'\nFeatures: {feature_names}')
    logger.info(f'df.shape {df.shape}, x.shape {x.shape}')

    # ### models
    # load pretrained model
    logger.info("Loading the best pretrained model...")
    model = pickle.load(open(f'../models/{args.pretrained}.sav', 'rb'))
    # predict
    logger.info('Predicting...')
    preds = model.predict(x)
    # save predictions
    logger.info('Saving predictions...')
    pred_name = f'../preds/{args.tile_id}.tiff'
    save_predictions_geotiff(meta, preds, pred_name)
    logger.info(f'Predictions are saved to {pred_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str,
                        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/',
                        help='Base directory to all the images.')
    parser.add_argument('--tile_id', type=str, default='43SFR')
    parser.add_argument('--pretrained', type=str, help='Filename of the best pretrained models.')

    parser.add_argument('--vis_ts', type=bool, default=False)
    parser.add_argument('--vis_profile', type=bool, default=False)
    parser.add_argument('--feature_engineering', type=bool, default=False)
    parser.add_argument('--scaling', type=str, default=None, choices=[None, 'standardize', 'normalize'])

    args = parser.parse_args()

    cropland_predict(args)
