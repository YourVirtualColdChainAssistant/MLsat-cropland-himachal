import argparse
import datetime
import numpy as np

from src.utils.util import get_log_dir, get_logger, get_grid_idx, count_classes, get_cropland_mask
from src.data.prepare_data import Pipeline, train_test_split, get_spatial_cv_fold, get_unlabeled_data
from src.models.base_model import ModelCropSpecific
from src.evaluation.evaluate import visualize_train_test_grid_split


def classifier_apples(args):
    # logger
    now = datetime.datetime.now().strftime("%m%d-%H%M%S")
    log_time = args.base_name.split('_')[0]
    base_model_name = args.base_name.split('_')[1]
    logger = get_logger(get_log_dir(), __name__,
                        f'{now}_apples_w_{args.base_name}.log', level='INFO')
    logger.info(args)
    logger.info('----- Crop-specific Classification -----')
    from_dir = args.images_dir + 'clip_new_labels/'

    # follow pipeline
    pipe = Pipeline(logger, from_dir)
    df = pipe.pipeline()
    meta = pipe.meta
    num_feature = df.columns.shape[0] - 1
    x = df.iloc[:, :num_feature].values

    # add new columns that are useful for Apples Classification
    if args.dataset_split_by == 'spatial':
        logger.info('Getting the grid idx...')
        df['grid_idx'] = list(get_grid_idx(args.grid_size, meta['height'], meta['width']).reshape(-1))
        logger.info('  ok')
    logger.info('Getting cropland mask...')
    cropland_mask = get_cropland_mask(df, num_feature, args.base_name)
    count_classes(logger, cropland_mask)
    logger.info('  ok')

    # train val test split
    if args.train_from == 'cropland':
        mask_pos = df.label.values == 1
        if args.dataset_split_by == 'spatial':
            logger.info('Spatial train-val-test split...')
            # get positive data as 1
            x_train_val_pos, x_test, y_train_val_pos, y_test, grid_idx_train_val_pos, grid_idx_test = \
                train_test_split(logger, df, mask_pos, num_feature, split_by='spatial',
                                 binary=False, test_ratio=0.2, random_seed=args.random_seed)
            y_train_val_pos = np.ones_like(y_train_val_pos, dtype=int)
            # add unlabeled data as 0
            mask_unl = df.label.values == 2
            x_train_val_unl, y_train_val_unl, grid_idx_train_val_unl = \
                get_unlabeled_data(df, mask_unl, num_feature)
            # concatenate PU data
            x_train_val_pu = np.concatenate([x_train_val_pos, x_train_val_unl], axis=0)
            y_train_val_pu = np.concatenate([y_train_val_pos, y_train_val_unl], axis=0)
            grid_idx_train_val_pu = np.concatenate([grid_idx_train_val_pos, grid_idx_train_val_unl], axis=0)
            logger.info('labeled vs no_label classes:')
            count_classes(logger, y_train_val_pu)
            # use pos/PU data to do spatial cv
            data_cv_pos, grid_idx_fold_pos = \
                get_spatial_cv_fold(x_train_val_pos, y_train_val_pos, grid_idx_train_val_pos)
            data_cv_pu, grid_idx_fold_pu = \
                get_spatial_cv_fold(x_train_val_pu, y_train_val_pu, grid_idx_train_val_pu)
            visualize_train_test_grid_split(meta, args.grid_size, grid_idx_test, [grid_idx_train_val_pu],
                                            f'../figs/apples_{args.grid_size}_train_test_split_{args.random_seed}.tiff')
            logger.info('Saved train-test evaluation.')
            visualize_train_test_grid_split(meta, args.grid_size, grid_idx_test, grid_idx_fold_pu,
                                            f'../figs/apples_{args.grid_size}_train_val_test_split_{args.random_seed}.tiff')
            logger.info('Saved train-val-test evaluation.')

    # ### models
    # ## OC-SVM
    ocsvm = ModelCropSpecific(logger, log_time, 'ocsvm')
    # # choose from
    # grid search
    # if args.dataset_split_by == 'random':
    #     ocsvm.find_best_parameters(x_train_val_pos, y_train_val_pos, search_by=args.cv_search_by)
    # else:
    #     ocsvm.find_best_parameters(x_train_val_pos, y_train_val_pos, search_by=args.cv_search_by,
    #                                cv=data_cv_pos)
    # ocsvm.fit_and_save_best_model(x_train_val_pos, y_train_val_pos)
    # fit known best parameters
    ocsvm.fit_and_save_best_model(x_train_val_pos, y_train_val_pos, {'gamma': 'scale', 'kernel': 'rbf', 'nu': 0.5})
    # predict and evaluation
    ocsvm.evaluate(x_test, y_test)
    ocsvm.predict_and_save(x, meta)  # predict from scratch
    ocsvm.predict_and_save(x, meta, cropland_mask)

    # ## PUL
    pul = ModelCropSpecific(logger, log_time, 'pul')
    # # choose from
    # grid search
    if args.dataset_split_by == 'spatial':
        pul.find_best_parameters(x_train_val_pu, y_train_val_pu, search_by=args.cv_search_by, cv=data_cv_pu)
    else:
        pul.find_best_parameters(x_train_val_pu, y_train_val_pu, search_by=args.cv_search_by)
    pul.fit_and_save_best_model(x_train_val_pu, y_train_val_pu)
    # fit known best parameters
    # pul.fit_and_save_best_model(x_train_val_pu, y_train_val_pu, # TODO: argument)
    # # predict and evaluation
    pul.evaluate(x_test, y_test)
    pul.predict_and_save(x, meta)
    pul.predict_and_save(x, meta, cropland_mask)


# ##############################
# #### previous codes
# ##############################

# add a column indicating stage 1 labels
# df['label_stage1'] = df['label'].values.copy()
# df.loc[df.label.values == 2, 'label_stage1'] = 1
#
# # prepare x and y
# x = df.iloc[:, :-2].values
# feature_num = df.columns[:-2].shape[0]
# logger.info('--- Print data summary ---')
# logger.info(f'\n{feature_num} features: {df.columns[:-2]}')
# logger.info(f'  x.shape {x.shape}')
#
# # load pretrained model
#
# logger.info(f"--- {base_model_name.upper()} ---")
# logger.info(f"Loading pretrained {base_model_name.upper()}...")
# clf = pickle.load(open(f'../models/{args.pretrained_model_name}.sav', 'rb'))
# logger.info('Predicting x ...')
# clf_preds = clf.predict(x)
#
# # ### prepare new label
# df['pred_stage1'] = clf_preds
# # add crops labels
# df['crops'] = df.label_stage1.values
# df.loc[df.pred_stage1.values == 0, 'crops'] = clf_preds[df.pred_stage1.values == 0]
# # add apples labels
# df['apples'] = -1
# df.loc[df.label.values == 1, 'apples'] = 1
# # get x and y for apples vs. other crops classification
# x_crops = df.iloc[df.crops.values > 0, :feature_num].values
# y_crops = df.loc[df.crops.values > 0, 'apples'].values

# logger.info('--- OC-SVM ---')
# ocsvm = OneClassSVM(gamma='auto').fit(x_crops)  # initialize and fit
# logger.info('Predicting x_crops...')
# ocsvm_pred = ocsvm.predict(x_crops)  # predict
# df['pred_ocsvm'] = -1  # merge into df
# df.loc[df.crops.values > 0, 'pred_ocsvm'] = ocsvm_pred
# logger.info('Saving OC-SVM predictions...')
# ocsvm_pred_name = f'../preds/{log_time}_ocsvm.tif'
# save_predictions_geotiff(meta, df.pred_ocsvm.values, ocsvm_pred_name)
# logger.info(f'OC-SVM predictions are saved to {ocsvm_pred_name}')
#
# # PU Learning
# logger.info('--- PUL ---')
# svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
# pu = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
# pu.fit(x_crops, y_crops)
# logger.info('Fitted x_crops ...')
# pu_pred = pu.predict(x_crops)
# df['pred_pu'] = -1
# df.loc[df.crops.values > 0, 'pred_pu'] = pu_pred
# logger.info('Predicted x_crops ...')
# pu_model = f'../models/{log_time}_pu.sav'
# pickle.dump(pu, open(pu_model, 'wb'))
# logger.info(f'  Saved trained PUL to {pu_model}')
# logger.info('Saving PUL predictions...')
# pu_pred_name = f'../preds/{log_time}_pu.tif'
# save_predictions_geotiff(meta, df.pred_pu.values, pu_pred_name)
# logger.info(f'PUL predictions are saved to {pu_pred_name}')
#
# # PU Learning weighted
# logger.info('--- PUL weighted ---')
# svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
# pu_w = WeightedElkanotoPuClassifier(estimator=svc, labeled=10, unlabeled=20, hold_out_ratio=0.2)
# pu_w.fit(x_crops, y_crops)
# logger.info('Fitted x_crops ...')
# pu_w_pred = pu_w.predict(x_crops)
# df['pred_pu_w'] = -1
# df.loc[df.crops.values > 0, 'pred_pu_w'] = pu_w_pred
# logger.info('Predicted x_crops ...')
# pu_w_model = f'../models/{log_time}_pu_w.sav'
# pickle.dump(pu_w, open(pu_w_model, 'wb'))
# logger.info(f'  Saved trained PUL_W to {pu_w_model}')
# logger.info('Saving PUL_W predictions...')
# pu_w_pred_name = f'../preds/{log_time}_pu_w.tif'
# save_predictions_geotiff(meta, df.pred_pu_w.values, pu_w_pred_name)
# logger.info(f'PUL_W predictions are saved to {pu_w_pred_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str,
                        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/',
                        help='Base directory to all the images.')
    parser.add_argument('--base_name', type=str, default='1023-162137_rfc',
                        help='Filenames of pretrained models.')
    parser.add_argument('--dataset_split_by', type=str, default='spatial', choices=['random', 'spatial'],
                        help='Method to split train-val-test dataset.')
    parser.add_argument('--cv_search_by', type=str, default='grid', choices=['random', 'grid'],
                        help='Method to do cross validation.')
    parser.add_argument('--grid_size', type=int, default=64,
                        help='Size of grid during spatial split.')
    parser.add_argument('--random_seed', type=int, default=22,
                        help='Random see for train_test_split.')
    parser.add_argument('--train_from', type=str, default='cropland', choices=['scratch', 'cropland'],
                        help='Crop-specific classification can be trained either from scratch or cropland map.')
    args = parser.parse_args()
    classifier_apples(args)
