import pandas as pd
from skimage.draw import draw
import pickle
import argparse
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from util import *
from util import get_log_dir, get_logger
from visualization import NVDI_profile
from prepare_data import Pipeline


def test(args):
    # logger
    logger_filename = datetime.datetime.now().strftime("%m%d-%H%M%S")
    logger = get_logger(get_log_dir(), __name__,
                        f'{logger_filename}_test.log', level='INFO')
    logger.info(args)
    logger.info('----- testing -----')
    from_dir = args.images_dir + 'clip/'

    # merge all labels
    merge_shapefiles()
    logger.info('Merged all labels')

    # follow pipeline
    pipe = Pipeline(logger, from_dir)
    meta = pipe.meta
    df = pipe.pipeline()

    # prepare x and y
    x = df.iloc[:, :-1].values
    y_3class = df.label.values  # raw data with 3 classes
    y = y_3class.copy()
    if args.binary_classifier:
        # modify to 2 classes
        y[y == 2] = 1
    logger.info('--- Print data summary ---')
    logger.info('With 3 classes:')
    count_classes(logger, y_3class)
    if args.binary_classifier:
        logger.info('With 2 classes:')
        count_classes(logger, y)
    logger.info(f'\nFeatures: {df.columns[:-1]}')
    logger.info(f'  x.shape {x.shape}, y.shape {y.shape}')

    # ### models
    # SVM
    logger.info("--- SVM ---")
    # load pretrained model
    logger.info("Loading pretrained SVM...")
    svm = pickle.load(open(f'../models/{args.pretrained_models[0]}.sav', 'rb'))
    # predict
    logger.info('Predicting...')
    svm_preds = svm.predict(x)
    # save predictions
    logger.info('Saving SVM predictions...')
    svm_pred_name = f'../preds/{args.pretrained_models[0]}.tif'
    save_predictions_geotiff(meta, svm_preds, svm_pred_name)
    logger.info(f'SVM predictions are saved to {svm_pred_name}')

    # Random forest
    logger.info("--- Random Forest ---")
    # load pretraind model
    logger.info("Loading pretrained RFC...")
    rfc = pickle.load(open(f'../models/{args.pretrained_models[1]}.sav', 'rb'))
    # predict
    logger.info('Predicting...')
    rfc_preds = rfc.predict(x)
    # save predictions
    logger.info('Saving RFC predictions...')
    rfc_pred_name = f'../preds/{args.pretrained_models[1]}.tif'
    save_predictions_geotiff(meta, rfc_preds, rfc_pred_name)
    logger.info(f'RFC predictions are saved to {rfc_pred_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images_dir',
        type=str,
        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/',
        help='Base directory to all the images.'
    )
    parser.add_argument(
        '--binary_classifier',
        type=bool,
        default=True,
        help='Classify 2 or 3 classes.'
    )
    parser.add_argument(
        '--pretrained_models',
        type=list,
        default=['svm_2021-10-01-18-44-45', 'rfc_2021-10-01-18-44-45'],
        help='Filenames of pretrained models.'
    )
    args = parser.parse_args()
    test(args)