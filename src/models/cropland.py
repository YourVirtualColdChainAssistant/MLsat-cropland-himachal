import pickle
from numpy import meshgrid
from scipy.stats import uniform, loguniform, randint
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from src.evaluation.evaluate import evaluate_by_gfsad, evaluate_by_copernicus, \
    impurity_importance_table, permutation_importance_table

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectKBest, f_classif
from src.models.base_model import convert_partial_predictions
from src.utils.util import save_predictions_geotiff
from src.evaluation.evaluate import adjust_raster_size
        

def test(logger, model, x_test, y_test, meta, index, cat_mask, region_shp_path, 
        pred_name, color_by_height, feature_names=None, work_station=False):
    logger.info("## Testing")
    # predict
    y_test_pred = model.predict(x_test)
    y_test_pred_converted = convert_partial_predictions(y_test_pred, index, meta)
    # save prediction
    pred_path = f'./preds/{pred_name}.tiff'
    save_predictions_geotiff(y_test_pred_converted, meta, pred_path, region_shp_path, 
                            cat_mask, color_by_height)
    logger.info(f'Saved predictions to {pred_path}')
    # evaluate
    logger.info('Evaluating by metrics...')
    metrics = evaluate_by_metrics(y_test, y_test_pred)
    logger.info(f'\n{metrics}')
    logger.info('Evaluating by open datasets')
    msgs = evaluate_by_open_datasets(meta, region_shp_path, pred_name, label_only=True, work_station=work_station)
    for msg in msgs:
        logger.info(msg)
    if feature_names is not None:
        logger.info('Evaluating by feature importance...')
        evaluate_by_feature_importance(model, x_test, y_test, feature_names, pred_name)


def predict(logger, model, x, meta, cat_mask, region_shp_path, 
            pred_name, color_by_height, work_station=False):
    logger.info("## Predicting")
    y_pred = model.predict(x)
    # save prediction
    pred_path = f'./preds/{pred_name}.tiff'
    save_predictions_geotiff(y_pred, meta, pred_path, region_shp_path, cat_mask, color_by_height)
    logger.info(f'Saved predictions to {pred_path}')
    # evaluate 
    msgs = evaluate_by_open_datasets(meta, region_shp_path, pred_name, label_only=False, work_station=work_station)
    for msg in msgs:
        logger.info(msg)


def evaluate_by_metrics(y_test, y_test_pred):
    # !! `labels` is related to the discrete number
    return classification_report(y_test, y_test_pred, labels=[2, 3], target_names=['croplands', 'non-croplands'])


def evaluate_by_feature_importance(model, x_test, y_test, feature_names, pred_name):
    PI_path = f'./preds/{pred_name}_PI.csv'
    permutation_importance_table(model, x_test, y_test, feature_names, PI_path)
    if isinstance(model, RandomForestClassifier()):
        II_path = f'./preds/{pred_name}_II.csv'
        impurity_importance_table(feature_names, model.feature_importances_, II_path)


def evaluate_by_open_datasets(meta, region_shp_path, pred_name, label_only=True, work_station=False):
    k_drive = '2021-data-org/4. RESEARCH_n/ML/MLsatellite/Data/layers_india/ancilliary_data/'
    ancilliary_path = '/home/lida/DFS/Projects/' + k_drive if work_station else 'K:/' + k_drive
    pred_path = f'./preds/{pred_name}.tiff'
    district = region_shp_path.split('/')[-2].split('_')[-1]
    msgs = []

    gfsad_args = {
        'dataset': 'gfsad',
        'raw_path': ancilliary_path + 'cropland/GFSAD30/GFSAD30SAAFGIRCE_2015_N30E70_001_2017286103800.tif',
        'evaluate_func': evaluate_by_gfsad
    }
    copernicus_args = {
        'dataset': 'copernicus',
        'raw_path': ancilliary_path + 'landcover/Copernicus_LC100m/INDIA_2019/' + \
                    'E060N40_PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif',
        'evaluate_func': evaluate_by_copernicus
    }

    for ds in [gfsad_args, copernicus_args]:
        dataset, raw_path, evaluate_func = ds['dataset'], ds['raw_path'], ds['evaluate_func']
        out_path = f'./data/open_datasets/{dataset}_{district}.tiff'
        print(f'Comparing {dataset.upper()} dataset with predictions...')
        adjust_raster_size(raw_path, out_path, region_shp_path, meta, label_only)
        msgs.append(evaluate_func(pred_path, out_path))
    return msgs


def get_model_and_params_dict_grid(model_name, random_state, testing, study_scaling):
    if model_name == 'svc':
        model = SVC()
        if not testing:
            params_dict = dict(
                classification__C=[0.5, 1, 10, 100],
                classification__gamma=['scale', 'auto'],
                classification__kernel=['poly', 'rbf'],
                classification__random_state=[random_state]
            )
        else:
            params_dict = dict(
                classification__C=[1],
                classification__gamma=['scale'],
                classification__kernel=['rbf'],
                classification__random_state=[random_state]
            )
    elif model_name == 'rfc':
        model = RandomForestClassifier()
        if not testing:
            params_dict = dict(
                classification__n_estimators=[100, 300, 500],
                classification__criterion=['gini', 'entropy'],
                classification__max_depth=[5, 10, 15],
                classification__max_samples=[0.5, 0.8, 1],
                classification__random_state=[random_state]
            )
        else:
            params_dict = dict(
                classification__n_estimators=[100],
                classification__criterion=['gini'],
                classification__max_depth=[5],
                classification__max_samples=[0.8],
                classification__random_state=[random_state]
            )
    else:  # self.model_name == 'mlp':
        model = MLPClassifier()
        if not testing:
            params_dict = dict(
                classification__hidden_layer_sizes=[(100,), (300,), (300, 300)],
                classification__alpha=[0.0001, 0.0005, 0.001, 0.005],
                classification__max_iter=[200, 500],
                classification__activation=['relu'],
                classification__early_stopping=[True],
                classification__random_state=[random_state]
            )
        else:
            params_dict = dict(
                classification__hidden_layer_sizes=[(100,)],
                classification__alpha=[0.0001],
                classification__max_iter=[200],
                classification__activation=['relu'],
                classification__early_stopping=[True],
                classification__random_state=[random_state]
            )
    if study_scaling:
        params_list = [
            dict(params_dict.items() + {'scale_minmax': ['passthrough']}.items()),
            dict(params_dict.items() + {'scale_std': ['passthrough']}.items())
        ]
    else:  # not study_scaling 
        params_list = [params_dict]
    return model, params_list


def get_best_model_initial(model_name, best_params):
    if model_name == 'svc':
        best_model = SVC(
            C=best_params['classification__C'],
            gamma=best_params['classification__gamma'],
            kernel=best_params['classification__kernel'],
            random_state=best_params['classification__random_state']
        )
    elif model_name == 'rfc':
        best_model = RandomForestClassifier(
            n_estimators=best_params['classification__n_estimators'],
            criterion=best_params['classification__criterion'],
            max_depth=best_params['classification__max_depth'],
            max_samples=best_params['classification__max_samples'],
            random_state=best_params['classification__random_state']
        )
    else:  # model_name == 'mlp':
        best_model = MLPClassifier(
            hidden_layer_sizes=best_params['classification__hidden_layer_sizes'],
            alpha=best_params['classification__alpha'],
            max_iter=best_params['classification__max_iter'],
            activation=best_params['classification__activation'],
            early_stopping=best_params['classification__early_stopping'],
            random_state=best_params['classification__random_state']
        )
    return best_model


def get_pipeline(model, scaling, study_scaling=False, engineer_feature=None):
    # decide pipeline structure
    pipeline_list = []
    if study_scaling:
        pipeline_list.append(('scale_minmax', MinMaxScaler()))
        pipeline_list.append(('scale_std', StandardScaler()))
    else:
        if scaling == 'standardize':
            pipeline_list.append(('scale_std', StandardScaler()))
        elif scaling == 'normalize':
            pipeline_list.append(('scale_minmax', MinMaxScaler()))
    if engineer_feature == 'select':
        pipeline_list.append(('feature_selection', SelectKBest(f_classif, k=10)))
    pipeline_list.append(('classification', model))
    # build pipeline
    pipe = Pipeline(pipeline_list)
    return pipe
