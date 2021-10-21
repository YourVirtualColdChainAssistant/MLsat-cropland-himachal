import pickle
import argparse
from sklearn.svm import SVC, OneClassSVM
from util import *
from util import get_log_dir, get_logger
from prepare_data import Pipeline
from pulearn import ElkanotoPuClassifier, WeightedElkanotoPuClassifier


def train_s2(args):
    # logger
    logger_filename = args.pretrained_model_name.split('_')[0]
    pretrained_model = args.pretrained_model_name.split('_')[1]
    logger = get_logger(get_log_dir(), __name__,
                        f'{logger_filename}_apples_on_{pretrained_model}.log', level='INFO')
    logger.info(args)
    logger.info('----- Crop-specific Classification -----')
    from_dir = args.images_dir + 'clip/'

    # follow pipeline
    pipe = Pipeline(logger, from_dir)
    meta = pipe.meta
    df = pipe.pipeline()
    # add a column indicating stage 1 labels
    df['label_stage1'] = df['label'].values.copy()
    df.loc[df.label.values == 2, 'label_stage1'] = 1

    # prepare x and y
    x = df.iloc[:, :-2].values
    feature_num = df.columns[:-2].shape[0]
    logger.info('--- Print data summary ---')
    logger.info(f'\n{feature_num} features: {df.columns[:-2]}')
    logger.info(f'  x.shape {x.shape}')

    # load pretrained model

    logger.info(f"--- {pretrained_model.upper()} ---")
    logger.info(f"Loading pretrained {pretrained_model.upper()}...")
    clf = pickle.load(open(f'../models/{args.pretrained_model_name}.sav', 'rb'))
    logger.info('Predicting x ...')
    clf_preds = clf.predict(x)

    # ### prepare new label
    df['pred_stage1'] = clf_preds
    # add crops labels
    df['crops'] = df.label_stage1.values
    df.loc[df.pred_stage1.values == 0, 'crops'] = clf_preds[df.pred_stage1.values == 0]
    # add apples labels
    df['apples'] = -1
    df.loc[df.label.values == 1, 'apples'] = 1
    # get x and y for apples vs. other crops classification
    x_crops = df.iloc[df.crops.values > 0, :feature_num].values
    y_crops = df.loc[df.crops.values > 0, 'apples'].values

    # ### apple vs. other crops
    # OC-SVM
    logger.info('--- OC-SVM ---')
    ocsvm = OneClassSVM(gamma='auto').fit(x_crops)  # initialize and fit
    logger.info('Predicting x_crops...')
    ocsvm_pred = ocsvm.predict(x_crops)  # predict
    df['pred_ocsvm'] = -1  # merge into df
    df.loc[df.crops.values > 0, 'pred_ocsvm'] = ocsvm_pred
    logger.info('Saving OC-SVM predictions...')
    ocsvm_pred_name = f'../preds/{logger_filename}_ocsvm.tif'
    save_predictions_geotiff(meta, df.pred_ocsvm.values, ocsvm_pred_name)
    logger.info(f'OC-SVM predictions are saved to {ocsvm_pred_name}')

    # PU Learning
    logger.info('--- PUL ---')
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    pu = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
    pu.fit(x_crops, y_crops)
    logger.info('Fitted x_crops ...')
    pu_pred = pu.predict(x_crops)
    df['pred_pu'] = -1
    df.loc[df.crops.values > 0, 'pred_pu'] = pu_pred
    logger.info('Predicted x_crops ...')
    pu_model = f'../models/{logger_filename}_pu.sav'
    pickle.dump(pu, open(pu_model, 'wb'))
    logger.info(f'  Saved trained PUL to {pu_model}')
    logger.info('Saving PUL predictions...')
    pu_pred_name = f'../preds/{logger_filename}_pu.tif'
    save_predictions_geotiff(meta, df.pred_pu.values, pu_pred_name)
    logger.info(f'PUL predictions are saved to {pu_pred_name}')

    # PU Learning weighted
    logger.info('--- PUL weighted ---')
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    pu_w = WeightedElkanotoPuClassifier(estimator=svc, labeled=10, unlabeled=20, hold_out_ratio=0.2)
    pu_w.fit(x_crops, y_crops)
    logger.info('Fitted x_crops ...')
    pu_w_pred = pu_w.predict(x_crops)
    df['pred_pu_w'] = -1
    df.loc[df.crops.values > 0, 'pred_pu_w'] = pu_w_pred
    logger.info('Predicted x_crops ...')
    pu_w_model = f'../models/{logger_filename}_pu_w.sav'
    pickle.dump(pu_w, open(pu_w_model, 'wb'))
    logger.info(f'  Saved trained PUL_W to {pu_w_model}')
    logger.info('Saving PUL_W predictions...')
    pu_w_pred_name = f'../preds/{logger_filename}_pu_w.tif'
    save_predictions_geotiff(meta, df.pred_pu_w.values, pu_w_pred_name)
    logger.info(f'PUL_W predictions are saved to {pu_w_pred_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images_dir',
        type=str,
        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/',
        help='Base directory to all the images.'
    )
    parser.add_argument(
        '--pretrained_model_name',
        type=str,
        default='1008-183014_rfc',
        help='Filenames of pretrained models.'
    )
    args = parser.parse_args()
    train_s2(args)
