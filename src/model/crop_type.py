import pickle
import random
import numpy as np
import rasterio
from shapely.geometry import Point
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, classification_report
from pulearn import ElkanotoPuClassifier, WeightedElkanotoPuClassifier

from src.data.load import load_geotiff
from src.data.write import save_predictions_geotiff
from src.model.util import convert_partial_predictions


def test(logger, model, x_test, y_test, meta, index,
         pred_name, color_by_height, region_indicator=None):
    """
    Test the crop type model performance, usually use partial data.

    Parameters
    ----------
    logger
    model
    x_test: np.array
        shape (n_test, n_feature)
    y_test: np.array
        shape (n_test, )
    meta: dict
        meta information
    index: np.array
        shape (n_test, )
    pred_name: str
        filename to save prediction
    color_by_height: bool
        whether to add a band with colored mask
    region_indicator: str or rasterio.window.Window or None
        an indicator of the area to predict

    Returns
    -------
    None
    """
    logger.info("## Testing")
    # predict
    y_test_pred = model.predict(x_test)
    print(y_test_pred)
    y_test_pred_converted = convert_partial_predictions(y_test_pred, index, meta['height'] * meta['width'])
    # save prediction
    pred_path = f'./preds/{pred_name}.tiff'
    save_predictions_geotiff(y_test_pred_converted, save_path=pred_path, meta=meta,
                             region_indicator=region_indicator, color_by_height=color_by_height)
    logger.info(f'Saved predictions to {pred_path}')
    # evaluate
    logger.info('Evaluating by metrics...')
    metrics = evaluate_by_metrics(y_test, y_test_pred)
    logger.info(f'\n{metrics}')



def predict(logger, model, x, meta, cropland_mask,
            pred_path, color_by_height, region_indicator=None):
    """
    Make prediction of the crop type model, usually a whole area.

    Parameters
    ----------
    logger
    model
    x: np.array
        shape (n_data, n_feature)
    meta: dict
        meta data
    cropland_mask: None or np.array
        shape (n_data, )
    pred_path: str
        path to store predictions, usually at N drive
    color_by_height: bool
        whether to add a band with colored mask
    region_indicator: str or rasterio.window.Window or None
        an indicator of the area to predict

    Returns
    -------
    None
    """
    logger.info("## Predicting")
    if cropland_mask is None:  # predict from scratch
        y_pred = predict_from_scratch(logger, model, x)
    else:  # predict from cropland
        y_pred = predict_from_cropland(logger, model, x, cropland_mask)
    # save prediction
    save_predictions_geotiff(y_pred, save_path=pred_path, meta=meta,
                             region_indicator=region_indicator, color_by_height=color_by_height)
    logger.info(f'Saved predictions to {pred_path}')


def predict_from_cropland(logger, model, x, cropland_mask):
    logger.info('Predicting from cropland (with croplands only)...')
    x_cropland = x[cropland_mask]
    y_cropland_pred = model.predict(x_cropland)
    logger.info('Masking cropland region...')
    y_preds = convert_partial_predictions(y_cropland_pred, cropland_mask, x.size(0))
    logger.info('ok')
    return y_preds


def predict_from_scratch(logger, model, x):
    logger.info('Predicting from scratch (with all data)...')
    y_preds = model.predict(x)
    logger.info('ok')
    return y_preds


def get_cropland_mask(where, x):
    if where.endswith('.tiff'):
        arr, _ = load_geotiff(where)
        cropland_mask = arr[0] == 2
    elif where.endswith('.pkl'):
        estimator = pickle.load(open(where, 'rb'))
        pred = estimator.predict(x)
        cropland_mask = pred == 2
    return cropland_mask


def get_unlabeled_data(df, unlabeled_mask, n_feature):
    x_unl = df.iloc[unlabeled_mask, :n_feature].values
    y_unl = np.zeros(x_unl.shape[0], dtype=int)
    grid_idx_unl = df.loc[unlabeled_mask, 'grid_idx'].values
    return x_unl, y_unl, grid_idx_unl


def correct_predictions():
    pass


def resample_negatives(pos, neg):
    pass


def sample_unlabeled_idx(coords, grid, size, meta):
    iterable = iter([(feat, val) for feat, val in zip(grid.geometry, np.ones(grid.shape[0], dtype=int))])
    img = rasterio.features.rasterize(iterable, out_shape=(meta['height'], meta['width']),
                                      transform=meta['transform'])
    mask = img.reshape(-1) == 1

    valid_idx = coords.index[mask]

    return np.random.choice(valid_idx, size, replace=False)  



def evaluate_by_recall(y_test, y_test_pred):
    return recall_score(y_test, y_test_pred, average='macro')


def evaluate_by_accuracy(y_test, y_test_pred):
    return accuracy_score(y_test, y_test_pred, average='macro')


def evaluate_by_metrics(y_test, y_test_pred):
    # !! `labels` is related to the discrete number
    return classification_report(y_test, y_test_pred, labels=[2, 3], target_names=['apples', 'non-apples'])



def get_model_and_params_dict_grid(model_name, testing):
    """
    Get the initial crop type model and a dictionary of its hyper-parameter space.

    Parameters
    ----------
    model_name: str
        choices = [ocsvc, pul, pul-w]
    testing: bool
        whether we are testing or not

    Returns
    -------
    model:
        a sklearn model
    params_dict: dict
        a dictionary of hyper-parameter space
    """
    if model_name == 'ocsvm':
        model = OneClassSVM()
        if not testing:
            params_dict = dict(
                classification__kernel=['poly', 'rbf'],
                classification__gamma=['scale', 'auto', 0.4],
                classification__nu=[0.3, 0.5, 0.7]
            )
        else:
            params_dict = dict(
                classification__kernel=['rbf'],
                classification__gamma=['scale'],
                classification__nu=[0.5]
            )
    elif model_name == 'pul':
        model = ElkanotoPuClassifier(SVC(probability=True))
        if not testing:
            params_dict = dict(
                classification__estimator=[SVC(probability=True), RandomForestClassifier()],
                classification__hold_out_ratio=[0.1, 0.2]
            )
        else:
            params_dict = dict(
                classification__estimator=[SVC(probability=True)],
                classification__hold_out_ratio=[0.1]
            )
    else:  # model_name = 'pul-w'
        model = WeightedElkanotoPuClassifier(SVC(probability=True), labeled=10, unlabeled=20)
        if not testing:
            params_dict = dict(
                classification__estimator=[SVC(probability=True), RandomForestClassifier()],
                classification__labeled=[10, 15],
                classification__unlabeled=[20, 10],
                classification__hold_out_ratio=[0.1, 0.2]
            )
        else:
            params_dict = dict(
                classification__estimator=[SVC(probability=True)],
                classification__labeled=[10],
                classification__unlabeled=[20],
                classification__hold_out_ratio=[0.1]
            )

    return model, params_dict


def get_best_model_initial(model_name, best_params):
    """
    Get the defined crop type model given parameters.

    Parameters
    ----------
    model_name: str
        choices = [ocsvc, pul, pul-w]
    best_params: dict
        a dictionary of best hyper-parameters of the model

    Returns
    -------
    best_model: a sklearn model
    """
    if model_name == 'ocsvm':
        best_model = OneClassSVM(
            kernel=best_params['classification__kernel'],
            gamma=best_params['classification__gamma'],
            nu=best_params['classification__nu']
        )
    elif model_name == 'pul':
        best_model = ElkanotoPuClassifier(
            estimator=best_params['classification__estimator'],
            hold_out_ratio=best_params['classification__hold_out_ratio']
        )
    else:  # model_name = 'pul-w'
        best_model = WeightedElkanotoPuClassifier(
            estimator=best_params['classification__estimator'],
            labeled=best_params['classification__labeled'],
            unlabeled=best_params['classification__unlabeled'],
            hold_out_ratio=best_params['classification__hold_out_ratio']
        )
    return best_model
