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
from util import get_log_dir, get_logger, merge_shapefiles
from visualization import NVDI_profile
from prepare_data import Pipeline


def train(args):
    # logger
    logger_filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = get_logger(get_log_dir(), __name__,
                        f'{logger_filename}_train.log', level='INFO')
    logger.info(args)
    logger.info('----- training -----')
    from_dir = args.images_dir + 'clip/'

    # merge all labels
    merge_shapefiles()
    logger.info('Merged all labels')

    # check NDVI profile
    ndvi_profile = NVDI_profile(logger, from_dir, '../data/all-labels/all-labels.shp')
    ndvi_profile.raw_profile()
    ndvi_profile.weekly_profile()
    ndvi_profile.monthly_profile()

    # follow pipeline
    pipe = Pipeline(logger, from_dir)
    df = pipe.pipeline()
    # select those with labels (deduct whose labels are 0)
    df_train_val = df[df['label'] != 0].reset_index(drop=True)

    # prepare x and y
    x = df_train_val.iloc[:, :-1].values
    y_3class = df_train_val.label.values  # raw data with 3 classes
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

    # split train validation set
    x_train, x_val, y_train, y_val = \
        train_test_split(x, y, test_size=0.2, random_state=42)
    feature_names = df_train_val.columns[:-1]
    logger.info(f'\nFeatures: {feature_names}')
    logger.info(f'  x.shape {x.shape}, y.shape {y.shape}')
    logger.info(f'  x_train.shape {x_train.shape}, y_train.shape {y_train.shape}')
    logger.info(f'  x_val.shape {x_val.shape}, y_val.shape {y_val.shape}')

    # ### models
    # SVM
    logger.info("--- SVM ---")
    # train model
    svm = SVC()
    svm.fit(x_train, y_train)
    y_val_pred_svm = svm.predict(x_val)
    # report results
    logger.info(f"\n{classification_report(y_val, y_val_pred_svm, target_names=['crops', 'non-crops'])}")
    # save model
    svm_model = f'../models/svm_{logger_filename}.sav'
    pickle.dump(svm, open(svm_model, 'wb'))
    logger.info(f'  Saved pre-trained SVM to {svm_model}')
    # feature importance - PI
    svm_PI = f'../preds/svm_PI_{logger_filename}.csv'
    permutation_importance_table(svm, x_val, y_val, feature_names, f'{svm_PI}')
    logger.info(f'  Saved permutation importance to {svm_PI}')

    # Random forest
    logger.info("--- Random Forest ---")
    # train model
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    y_val_pred_rfc = rfc.predict(x_val)
    # report results
    logger.info(f"\n{classification_report(y_val, y_val_pred_rfc, target_names=['crops', 'non-crops'])}")
    # save model
    rfc_model = f'../models/rfc_{logger_filename}.sav'
    pickle.dump(rfc, open(rfc_model, 'wb'))
    logger.info(f'  Saved pre-trained RFC to {rfc_model}')
    # feature importance - II
    rfc_II = f'../preds/rfc_II_{logger_filename}.csv'
    impurity_importance_table(feature_names, rfc.feature_importances_, f'{rfc_II}')
    logger.info(f'  Saved impurity importance to {rfc_II}')
    # feature importance - PI
    rfc_PI = f'../preds/rfc_PI_{logger_filename}.csv'
    permutation_importance_table(rfc, x_val, y_val, feature_names, f'{rfc_PI}')
    logger.info(f'  Saved permutation importance to {rfc_PI}')


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
    args = parser.parse_args()
    train(args)
