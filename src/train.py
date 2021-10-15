from skimage.draw import draw
import pickle
import argparse
import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from util import *
from util import get_log_dir, get_logger, merge_shapefiles
from visualization import NVDI_profile
from prepare_data import Pipeline


def train(args):
    # logger
    logger_filename = datetime.datetime.now().strftime("%m%d-%H%M%S")
    logger = get_logger(get_log_dir(), __name__,
                        f'{logger_filename}_train.log', level='INFO')
    logger.info(args)
    logger.info('----- training -----')
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

    # select those with labels (deduct whose labels are 0)
    df_train_val_test = df[df['label'] != 0].reset_index(drop=True)
    x_train_val_test = df_train_val_test.iloc[:, :-1].values
    y_train_val_test_3c = df_train_val_test.label.values  # raw data with 3 classes
    y_train_val_test = y_train_val_test_3c.copy()
    if args.binary_classifier:
        # modify to 2 classes
        y_train_val_test[y_train_val_test == 2] = 1
    logger.info('--- Print data summary ---')
    logger.info('With 3 classes:')
    count_classes(logger, y_train_val_test_3c)
    if args.binary_classifier:
        logger.info('With 2 classes:')
        count_classes(logger, y_train_val_test)

    # split train & validation, test set
    x_train_val, x_test, y_train_val, y_test = \
        train_test_split(x_train_val_test, y_train_val_test, test_size=0.2, random_state=42)

    # print
    feature_names = df_train_val_test.columns[:-1]
    logger.info(f'\nFeatures: {feature_names}')
    logger.info(f'  x_train_val_test.shape {x_train_val_test.shape}, y_train_val_test.shape {y_train_val_test.shape}')
    logger.info(f'  x_train_val.shape {x_train_val.shape}, y_train_val.shape {y_train_val.shape}')
    logger.info(f'  x_test.shape {x_test.shape}, y_test.shape {y_test.shape}')

    # ### models
    # ## SVM
    logger.info("--- SVM ---")
    # train model
    svm_bs = SVC()
    # random search CV
    svm_params_dist = dict(
        C=[0.5, 1, 10, 100],
        kernel=['linear', 'poly', 'rbf']
    )
    svm_random = GridSearchCV(estimator=svm_bs, param_grid=svm_params_dist,
                              cv=3, verbose=3, n_jobs=-1)
    # # Fit the random search model
    # svm_random.fit(x_train_val, y_train_val)
    # svm_random_model = f'../models/{logger_filename}_svm_random.sav'
    # pickle.dump(svm_random, open(svm_random_model, 'wb'))
    # logger.info(f'  Saved SVM CV to {svm_random_model}')
    # # best parameter
    # logger.info(f"Best score {round(svm_random.best_score_, 4)} with best parameters: {svm_random.best_params_}")

    # fit the best model
    svm_random.best_params_ = {'C': 100, 'kernel': 'rbf'}
    logger.info(f"Fitting the best svm with {svm_random.best_params_}...")
    svm = SVC(C=svm_random.best_params_['C'],
              kernel=svm_random.best_params_['kernel'])
    svm.fit(x_train_val, y_train_val)
    logger.info("Predicting test data...")
    y_test_pred_svm = svm.predict(x_test)
    # report results
    logger.info(
        f"\n{classification_report(y_test, y_test_pred_svm, labels=[1, -1], target_names=['crops', 'non-crops'])}")
    # save model
    svm_model = f'../models/{logger_filename}_svm.sav'
    pickle.dump(svm, open(svm_model, 'wb'))
    logger.info(f'  Saved the best pre-trained SVM to {svm_model}')

    # reload the best model
    # logger.info("Loading pretrained SVM...")
    # svm = pickle.load(open('../models/1007-180844_svm.sav', 'rb'))

    # predict all
    logger.info('Predicting...')
    svm_preds = svm.predict(x)
    # save predictions
    logger.info('Saving SVM predictions...')
    svm_pred_name = f'../preds/{logger_filename}_svm.tif'
    save_predictions_geotiff(meta, svm_preds, svm_pred_name)
    logger.info(f'SVM predictions are saved to {svm_pred_name}')

    # feature importance - PI
    svm_PI = f'../preds/{logger_filename}_svm_PI.csv'
    permutation_importance_table(svm, x_test, y_test, feature_names, f'{svm_PI}')
    logger.info(f'  Saved permutation importance to {svm_PI}')

    # ## Random forest
    logger.info("--- RFC ---")
    # train model
    rfc_bs = RandomForestClassifier()

    # random search CV
    rfc_params_dist = dict(
        n_estimators=[100, 300, 500],
        criterion=['gini', 'entropy'],
        max_depth=[5, 10, 15],
        max_samples=[0.5, 0.8, 1]
    )
    # rfc_params_dist = dict(
    #     n_estimators=[100],
    #     criterion=['gini']
    # )
    rfc_random = GridSearchCV(estimator=rfc_bs, param_grid=rfc_params_dist,
                              cv=3, verbose=3, n_jobs=-1)
    # # Fit the random search model
    # rfc_random.fit(x_train_val, y_train_val)
    # rfc_random_model = f'../models/{logger_filename}_rfc_random.sav'
    # pickle.dump(rfc_random, open(rfc_random_model, 'wb'))
    # logger.info(f'  Saved RFC CV to {rfc_random_model}')
    # # best parameter
    # logger.info(f"Best score {round(rfc_random.best_score_, 2)} with best parameters: {rfc_random.best_params_}")

    # fit the best model
    rfc_random.best_params_ = {'n_estimators': 500, 'criterion': 'entropy', 'max_depth': 15, 'max_samples': 0.8}
    rfc = RandomForestClassifier(n_estimators=rfc_random.best_params_['n_estimators'],
                                 criterion=rfc_random.best_params_['criterion'],
                                 max_depth=rfc_random.best_params_['max_depth'],
                                 max_samples=rfc_random.best_params_['max_samples'])
    rfc.fit(x_train_val, y_train_val)
    y_test_pred_rfc = rfc.predict(x_test)
    # report results
    logger.info(f"\n{classification_report(y_test, y_test_pred_rfc, labels=[1,-1], target_names=['crops', 'non-crops'])}")
    # save model
    rfc_model = f'../models/{logger_filename}_rfc.sav'
    pickle.dump(rfc, open(rfc_model, 'wb'))
    logger.info(f'  Saved the best pre-trained RFC to {rfc_model}')

    # predict all
    logger.info('Predicting...')
    rfc_preds = rfc.predict(x)
    # save predictions
    logger.info('Saving RFC predictions...')
    rfc_pred_name = f'../preds/{logger_filename}_rfc.tif'
    save_predictions_geotiff(meta, rfc_preds, rfc_pred_name)
    logger.info(f'RFC predictions are saved to {rfc_pred_name}')

    # feature importance - II
    rfc_II = f'../preds/{logger_filename}_rfc_II.csv'
    impurity_importance_table(feature_names, rfc.feature_importances_, f'{rfc_II}')
    logger.info(f'  Saved impurity importance to {rfc_II}')

    # feature importance - PI
    rfc_PI = f'../preds/{logger_filename}_rfc_PI.csv'
    permutation_importance_table(rfc, x_test, y_test, feature_names, f'{rfc_PI}')
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
