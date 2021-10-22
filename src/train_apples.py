import pickle
import argparse
from util import get_log_dir, get_logger
from prepare_data import Pipeline, train_test_split, spatial_cross_validation
from models import ModelCropSpecific
from visualization import visualize_train_test_grid_split


def classifier_apples(args):
    # logger
    log_time = args.base_name.split('_')[0]
    base_model_name = args.base_name.split('_')[1]
    logger = get_logger(get_log_dir(), __name__,
                        f'{log_time}_apples_on_{base_model_name}.log', level='INFO')
    logger.info(args)
    logger.info('----- Crop-specific Classification -----')
    from_dir = args.images_dir + 'clip_new_labels/'

    # follow pipeline
    pipe = Pipeline(logger, from_dir)
    df = pipe.pipeline()
    meta = pipe.meta
    num_feature = df.columns.shape[0] - 1

    # add new columns that are useful for Apples Classification
    df['l_apples'] = 0
    df.loc[df.label.values == 1, 'label_apples'] = 1
    df['l_cropland'] = 0
    df.loc[(df.label.values == 1) | (df.label.values == 2), 'label_cropland'] = 1

    if args.train_from:
        train(df, meta, num_feature, (df.label == 1) | (df.label == 3), logger, log_time)
    else:
        train(df, meta, num_feature, df.label == 1, logger, log_time)


def train(df, meta, num_feature, train_val_test_mask, logger, log_time):
    # train val test split
    if args.dataset_split == 'random':
        logger.info('Random train-val-test split...')
        x_train_val, x_test, y_train_val, y_test, grid_idx_train_val, grid_idx_test = \
            train_test_split(logger, df, train_val_test_mask, num_feature, split_by='random',
                             spatial_dict=None, test_ratio=0.2, random_seed=args.random_seed)
    elif args.dataset_split == 'spatial':
        logger.info('Spatial train-val-test split...')
        spatial_dict = {'grid_size': args.grid_size, 'height': meta['height'], 'width': meta['width']}
        x_train_val, x_test, y_train_val, y_test, grid_idx_train_val, grid_idx_test = \
            train_test_split(logger, df, train_val_test_mask, num_feature, split_by='spatial',
                             spatial_dict=spatial_dict, test_ratio=0.2, random_seed=args.random_seed)
        data_cv, grid_idx_fold = spatial_cross_validation(x_train_val, y_train_val, grid_idx_train_val)
        visualize_train_test_grid_split(meta, spatial_dict, grid_idx_test, [grid_idx_train_val],
                                        f'../preds/apples_train_test_split_{args.random_seed}.tiff')
        logger.info('Saved train-test visualization.')
        visualize_train_test_grid_split(meta, spatial_dict, grid_idx_test, grid_idx_fold,
                                        f'../preds/apples_train_val_test_split_{args.random_seed}.tiff')
        logger.info('Saved train-val-test visualization.')

    # ### models
    # ## OC-SVM
    ocsvm = ModelCropSpecific(logger, log_time, 'ocsvm')
    # # choose from
    # grid search
    if args.dataset_split == 'random':
        ocsvm.find_best_parameters(x_train_val, y_train_val, search_by=args.cv_search_by)
    else:
        ocsvm.find_best_parameters(x_train_val, y_train_val, search_by=args.cv_search_by, cv=data_cv)
    ocsvm.fit_and_save_best_model(x_train_val, y_train_val)
    # fit known best parameters
    # svc.fit_and_save_best_model(x_train_val, y_train_val, {'C': 100, 'kernel': 'rbf'})
    # reload pretrained model
    # ocsvm.load_pretrained_model('../models/1021-181246_ocsvm.sav')
    # # predict and evaluate
    # svc.save_predictions(x, meta)
    ocsvm.evaluate(x_test, y_test)

    # ##############################
    # #### previous codes
    # ##############################

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

    logger.info(f"--- {base_model_name.upper()} ---")
    logger.info(f"Loading pretrained {base_model_name.upper()}...")
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

    logger.info('--- OC-SVM ---')
    ocsvm = OneClassSVM(gamma='auto').fit(x_crops)  # initialize and fit
    logger.info('Predicting x_crops...')
    ocsvm_pred = ocsvm.predict(x_crops)  # predict
    df['pred_ocsvm'] = -1  # merge into df
    df.loc[df.crops.values > 0, 'pred_ocsvm'] = ocsvm_pred
    logger.info('Saving OC-SVM predictions...')
    ocsvm_pred_name = f'../preds/{log_time}_ocsvm.tif'
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
    pu_model = f'../models/{log_time}_pu.sav'
    pickle.dump(pu, open(pu_model, 'wb'))
    logger.info(f'  Saved trained PUL to {pu_model}')
    logger.info('Saving PUL predictions...')
    pu_pred_name = f'../preds/{log_time}_pu.tif'
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
    pu_w_model = f'../models/{log_time}_pu_w.sav'
    pickle.dump(pu_w, open(pu_w_model, 'wb'))
    logger.info(f'  Saved trained PUL_W to {pu_w_model}')
    logger.info('Saving PUL_W predictions...')
    pu_w_pred_name = f'../preds/{log_time}_pu_w.tif'
    save_predictions_geotiff(meta, df.pred_pu_w.values, pu_w_pred_name)
    logger.info(f'PUL_W predictions are saved to {pu_w_pred_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str,
                        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/',
                        help='Base directory to all the images.')
    parser.add_argument('--base_name', type=str, default='1008-183014_rfc',
                        help='Filenames of pretrained models.')
    parser.add_argument('--dataset_split_by', type=str, default='spatial', choices=['random', 'spatial'],
                        help='Method to split train-val-test dataset.')
    parser.add_argument('--cv_search_by', type=str, default='grid', choices=['random', 'grid'],
                        help='Method to do cross validation.')
    parser.add_argument('--grid_size', type=int, default=64,
                        help='Size of grid during spatial split.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random see for train_test_split.')
    parser.add_argument('--train_from', type=str, default='scratch', choices=['scratch', 'cropland'],
                        help='Crop-specific classification can be trained either from scratch or cropland map.')
    args = parser.parse_args()
    classifier_apples(args)
