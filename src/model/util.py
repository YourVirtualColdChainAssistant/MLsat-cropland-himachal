import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif


def load_pretrained_model(self, pretrained_name):
    self._logger.info(f"Loading pretrained {pretrained_name}...")
    self.model = pickle.load(open(pretrained_name, 'rb'))
    self._logger.info('  ok')


def convert_partial_predictions(preds, index, size):
    y_preds = np.zeros(size)
    y_preds[index] = preds
    return y_preds


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
        pipeline_list.append(('feature_selection', SelectKBest(f_classif)))
    pipeline_list.append(('classification', model))
    # build pipeline
    pipe = Pipeline(pipeline_list)
    return pipe


def get_addtional_params(params_dict, testing, study_scaling, engineer_feature):
    feature_k_list = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600] if not testing else [10]
    if study_scaling and engineer_feature == 'select':
        p1 = params_dict.copy().update(
            {'scale_minmax': ['passthrough'], 'feature_selection__k': feature_k_list})
        p2 = params_dict.copy().update(
            {'scale_std': ['passthrough'], 'feature_selection__k': feature_k_list})
        params_list = [p1, p2]
    elif study_scaling and engineer_feature != 'select':
        p1 = params_dict.copy().update({'scale_minmax': ['passthrough']})
        p2 = params_dict.copy().update({'scale_std': ['passthrough']})
        params_list = [p1, p2]
    elif not study_scaling and engineer_feature == 'select':
        params_dict.update({'feature_selection__k': feature_k_list})
        params_list = [params_dict]
    else:  # not study_scaling and engineer_feature != 'select':
        params_list = [params_dict]
    return params_list
