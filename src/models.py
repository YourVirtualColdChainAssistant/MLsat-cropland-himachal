import pickle
import numpy as np
from scipy.stats import uniform, loguniform, randint
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, recall_score
from util import save_predictions_geotiff, save_cv_results
from evaluation import align_raster, clip_open_datasets_based_on_study_area, \
    compare_predictions_with_gfsad, compare_predictions_with_copernicus, \
    impurity_importance_table, permutation_importance_table
from pulearn import ElkanotoPuClassifier, WeightedElkanotoPuClassifier


class Model(object):
    def __init__(self, logger, log_time, model_name):
        self._logger = logger
        self._log_time = log_time
        self.model_name = model_name
        self.to_name = f'{self._log_time}_{self.model_name}'
        self.best_params = None
        self.model = None
        self._logger.info(f'===== {self.model_name.upper()} =====')

    def _check_model_name(self, model_list):
        if self.model_name not in model_list:
            raise ValueError(f'No such model {self.model_name}. Please choose from {model_list}.')

    def find_best_parameters(self, x_train_val, y_train_val, scoring=None, search_by='grid', cv=3, n_iter=10,
                             testing=False):
        self._logger.info(f"Finding the best parameters using {search_by} search...")
        if search_by == 'grid':
            model_base, model_params = self._get_model_base_and_params_list_grid(testing)
            model_search = GridSearchCV(estimator=model_base, param_grid=model_params, cv=cv, scoring=scoring,
                                        verbose=3, n_jobs=-1)
        elif search_by == 'random':
            if testing:
                n_iter = 1
            model_base, model_params = self._get_model_base_and_params_list_random()
            model_search = RandomizedSearchCV(estimator=model_base, param_distributions=model_params, cv=cv,
                                              scoring=scoring, verbose=3, n_jobs=-1, n_iter=n_iter)
        else:
            raise ValueError(f'No {search_by} search. Choose from ["grid", "random"]')

        model_search.fit(x_train_val, y_train_val)
        self._logger.info('  ok')
        # model_search_name = f'../models/{self.to_name}_{search_by}.csv'
        # save_cv_results(model_search.cv_results_, model_search_name)
        # self._logger.info(f'  Saved {self.model_name.upper()} {search_by} search results to {model_search_name}')
        self.best_params = model_search.best_params_
        self._logger.info(f"  Best score {model_search.best_score_:.4f} with best parameters: {self.best_params}")

    def fit_and_save_best_model(self, x_train_val, y_train_val, best_params=None):
        if best_params is not None:
            self.best_params = best_params
        if self.best_params is None:
            raise ValueError('No best parameters are specified.')
        self._logger.info(f"Fitting the best {self.model_name.upper()} with {self.best_params}...")
        self.model = self._get_best_model()
        if self.model_name == 'ocsvm':
            self.model.fit(x_train_val)
        else:
            self.model.fit(x_train_val, y_train_val)
        self._logger.info('  ok')
        model_name = f'../models/{self.to_name}.pkl'
        pickle.dump(self.model, open(model_name, 'wb'))
        self._logger.info(f'  Saved the best {self.model_name.upper()} to {model_name}')

    def predict_and_save(self, x, meta, to_name=None):
        self._logger.info('Predicting all the data...')
        y_preds = self.model.predict(x)
        self._logger.info('  ok')
        self._save_predictions(meta, y_preds, to_name)

    def load_pretrained_model(self, pretrained_name):
        self._logger.info(f"Loading pretrained {pretrained_name}...")
        self.model = pickle.load(open(pretrained_name, 'rb'))
        self._logger.info('  ok')

    def _save_predictions(self, meta, y_preds, to_name=None):
        if to_name is None:
            to_name = self.to_name
        preds_name = f'../preds/{to_name}.tiff'
        save_predictions_geotiff(meta, y_preds, preds_name)
        self._logger.info(f'  Saved {self.model_name.upper()} predictions to {preds_name}')

    def _get_model_base_and_params_list_grid(self, testing):
        raise NotImplementedError

    def _get_model_base_and_params_list_random(self):
        raise NotImplementedError

    def _get_best_model(self):
        raise NotImplementedError


class ModelCropland(Model):
    def __init__(self, logger, log_time, model_name, pretrained_name=None):
        # inherent from parent
        super().__init__(logger, log_time, model_name)
        self._check_model_name(model_list=['svc', 'rfc', 'mlp', 'gru'])
        # load pretrained
        if pretrained_name is not None:
            self._logger.info(f'Changed log time from {log_time} to {pretrained_name.split("_")[0]}')
            self._log_time = pretrained_name.split('_')[0]
            if pretrained_name.split('_')[1] != self.model_name:
                raise ValueError('Initialized model is not the same as the pretrained model.')
            self.load_pretrained_model(f'../models/{pretrained_name}.pkl')

    def find_best_parameters(self, x_train_val, y_train_val, scoring=None, search_by='grid', cv=3, n_iter=10,
                             testing=False):
        super().find_best_parameters(x_train_val, y_train_val, scoring=scoring, search_by=search_by, cv=cv,
                                     n_iter=n_iter, testing=testing)

    def evaluate_by_metrics(self, x_test, y_test):
        self._logger.info('Evaluating by metrics...')
        self._logger.info("  Predicting test data...")
        y_test_pred = self.model.predict(x_test)
        self._logger.info('  ok')
        # !! `labels` is related to the discrete number
        self._logger.info(
            f"\n{classification_report(y_test, y_test_pred, labels=[2, 3], target_names=['crops', 'non-crops'])}"
        )

    def evaluate_by_feature_importance(self, x_test, y_test, feature_names):
        self._logger.info('Evaluating by feature importance...')
        model_PI = f'../preds/{self.to_name}_PI.csv'
        permutation_importance_table(self.model, x_test, y_test, feature_names, f'{model_PI}')
        self._logger.info('  ok')
        self._logger.info(f'  Saved permutation importance to {model_PI}')
        if self.model_name == 'rfc':
            model_II = f'../preds/{self.to_name}_II.csv'
            impurity_importance_table(feature_names, self.model.feature_importances_, f'{model_II}')
            self._logger.info(f'  Saved impurity importance to {model_II}')

    def evaluate_by_open_datasets(self, pred_path):
        self._logger.info('Evaluating by open datasets...')
        ancilliary_path = 'K:/2021-data-org/4. RESEARCH_n/ML/MLsatellite/Data/layers_india/ancilliary_data/'

        # gfsad
        gfsad_path = ancilliary_path + 'cropland/GFSAD30/GFSAD30SAAFGIRCE_2015_N30E70_001_2017286103800.tif'
        gfsad_clip_path = '../data/gfsad_clipped.tiff'
        gfsad_align_path = '../data/gfsad_aligned.tiff'

        self._logger.info('Comparing GFSAD dataset with predictions...')
        clip_open_datasets_based_on_study_area(gfsad_path, gfsad_clip_path)
        self._logger.info('  Clipped GFSAD dataset to study_region.')
        align_raster(pred_path, gfsad_clip_path, gfsad_align_path)
        self._logger.info('  Aligned GFSAD dataset to predictions.')
        compare_predictions_with_gfsad(pred_path, gfsad_align_path, self._logger)

        # copernicus
        copernicus_path = ancilliary_path + 'landcover/Copernicus_LC100m/INDIA_2019/' + \
                          'E060N40_PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif'
        copernicus_clip_path = '../data/copernicus_clipped.tiff'
        copernicus_align_path = '../data/copernicus_aligned.tiff'

        self._logger.info('Comparing Copernicus dataset with predictions...')
        clip_open_datasets_based_on_study_area(copernicus_path, copernicus_clip_path)
        self._logger.info('  Clipped Copernicus dataset to study_region.')
        align_raster(pred_path, copernicus_clip_path, copernicus_align_path)
        self._logger.info('  Aligned Copernicus dataset to predictions.')
        compare_predictions_with_copernicus(pred_path, copernicus_align_path, self._logger)

    def predict_and_save(self, x, meta, to_name=None):
        super().predict_and_save(x, meta, to_name)
        pred_path = f'../preds/{self.to_name}.tiff' if to_name is None else f'../preds/{to_name}.tiff'
        self.evaluate_by_open_datasets(pred_path)

    def load_pretrained_model(self, pretrained_name):
        self._logger.info(f"Loading pretrained {pretrained_name}...")
        self.model = pickle.load(open(pretrained_name, 'rb'))
        self._logger.info('  ok')

    def _get_model_base_and_params_list_grid(self, testing):
        if self.model_name == 'svc':
            model_base = SVC()
            if not testing:
                model_params_list = dict(
                    C=[0.5, 1, 10, 100],
                    gamma=['scale', 'auto'],
                    kernel=['poly', 'rbf']
                )
            else:
                model_params_list = dict(
                    C=[1],
                    gamma=['scale'],
                    kernel=['rbf']
                )
        elif self.model_name == 'rfc':
            model_base = RandomForestClassifier()
            if not testing:
                model_params_list = dict(
                    n_estimators=[100, 300, 500],
                    criterion=['gini', 'entropy'],
                    max_depth=[5, 10, 15],
                    max_samples=[0.5, 0.8, 1]
                )
            else:
                model_params_list = dict(
                    n_estimators=[100],
                    criterion=['gini'],
                    max_depth=[5],
                    max_samples=[0.8]
                )
        elif self.model_name == 'mlp':
            model_base = MLPClassifier()
            if not testing:
                model_params_list = dict(
                    hidden_layer_sizes=[(100,), (300,), (300, 300)],
                    alpha=[0.0001, 0.0005, 0.001, 0.005],
                    max_iter=[200, 500],
                    activation=['relu'],
                    early_stopping=[True]
                )
            else:
                model_params_list = dict(
                    hidden_layer_sizes=[(100,)],
                    alpha=[0.0001],
                    max_iter=[200],
                    activation=['relu'],
                    early_stopping=[True]
                )
        else:
            model_base = None
            model_params_list = None
        self._logger.info(f'  model base {model_base}')
        self._logger.info(f'  model parameters list {model_params_list}')
        return model_base, model_params_list

    def _get_model_base_and_params_list_random(self):
        if self.model_name == 'svc':
            model_base = SVC()
            model_params_dist = dict(
                C=loguniform(0.1, 100),
                gamma=['scale', 'auto'],
                kernel=['poly', 'rbf']
            )
        elif self.model_name == 'rfc':
            model_base = RandomForestClassifier()
            model_params_dist = dict(
                n_estimators=randint(100, 1000),
                criterion=['gini', 'entropy'],
                max_depth=randint(5, 15),
                max_samples=uniform(0.5, 0.5)  # uniform(loc, scale) -> a=loc, b=loc+scale
            )
        elif self.model_name == 'mlp':
            model_base = MLPClassifier()
            model_params_dist = dict(
                hidden_layer_sizes=[(100,), (300,)],
                alpha=loguniform(0.0001, 0.001),
                max_iter=randint(200, 500),
                activation=['relu'],
                early_stopping=[True]
            )
        else:
            model_base = None
            model_params_dist = None
        self._logger.info(f'  model base {model_base}')
        self._logger.info(f'  model parameters distribution {model_params_dist}')
        return model_base, model_params_dist

    def _get_best_model(self):
        if self.model_name == 'svc':
            model = SVC(
                C=self.best_params['C'],
                gamma=self.best_params['gamma'],
                kernel=self.best_params['kernel']
            )
        elif self.model_name == 'rfc':
            model = RandomForestClassifier(
                n_estimators=self.best_params['n_estimators'],
                criterion=self.best_params['criterion'],
                max_depth=self.best_params['max_depth'],
                max_samples=self.best_params['max_samples']
            )
        elif self.model_name == 'mlp':
            model = MLPClassifier(
                hidden_layer_sizes=self.best_params['hidden_layer_sizes'],
                alpha=self.best_params['alpha'],
                max_iter=self.best_params['max_iter'],
                activation=self.best_params['activation'],
                early_stopping=self.best_params['early_stopping']
            )
        else:
            model = None
        return model


class ModelCropSpecific(Model):
    def __init__(self, logger, log_time, model_name, pretrained_name=None):
        # inherent from parent
        super().__init__(logger, log_time, model_name)
        self._check_model_name(model_list=['ocsvm', 'pul', 'pul-w'])
        # add pretrained_name and base model
        if pretrained_name is None:
            self.base_model = None
            self._logger.info('Train from scratch.')
        else:
            self.load_pretrained_model(pretrained_name)

    def find_best_parameters(self, x_train_val, y_train_val, scoring='recall', search_by='grid', cv=3, testing=False):
        super().find_best_parameters(x_train_val, y_train_val, scoring=scoring, search_by=search_by,
                                     cv=cv, testing=testing)

    def evaluate_by_recall(self, x_test, y_test):
        self._logger.info('Evaluating by recall...')
        self._logger.info("  Predicting test data...")
        y_test_pred = self.model.predict(x_test)
        self._logger.info('  ok')
        self._logger.info(f"The best recall is {recall_score(y_test, y_test_pred, average='macro')}")

    def _get_model_base_and_params_list_grid(self, testing):
        if self.model_name == 'ocsvm':
            model_base = OneClassSVM()
            if not testing:
                model_params_list = dict(
                    kernel=['poly', 'rbf'],
                    gamma=['scale', 'auto', 0.4],
                    nu=[0.3, 0.5, 0.7]
                )
            else:
                model_params_list = dict(
                    kernel=['rbf'],
                    gamma=['scale'],
                    nu=[0.5]
                )
        elif self.model_name == 'pul':
            model_base = ElkanotoPuClassifier(SVC(probability=True))
            if not testing:
                model_params_list = dict(
                    estimator=[SVC(probability=True), RandomForestClassifier()],
                    hold_out_ratio=[0.1, 0.2]
                )
            else:
                model_params_list = dict(
                    estimator=[SVC(probability=True)],
                    hold_out_ratio=[0.1]
                )
        else:
            model_base = WeightedElkanotoPuClassifier(SVC(probability=True), labeled=10, unlabeled=20)
            if not testing:
                model_params_list = dict(
                    estimator=[SVC(probability=True), RandomForestClassifier()],
                    labeled=[10, 15],
                    unlabeled=[20, 10],
                    hold_out_ratio=[0.1, 0.2]
                )
            else:
                model_params_list = dict(
                    estimator=[SVC(probability=True)],
                    labeled=[10],
                    unlabeled=[20],
                    hold_out_ratio=[0.1]
                )
        self._logger.info(f'  model base {model_base}')
        self._logger.info(f'  model parameters list {model_params_list}')
        return model_base, model_params_list

    def _get_model_base_and_params_list_random(self):
        if self.model_name == 'ocsvm':
            model_base = OneClassSVM()
            model_params_dist = dict(
                kernel=['linear', 'poly', 'rbf'],
                gamma=uniform(0.2, 0.8),
                nu=uniform(0, 1)
            )
        elif self.model_name == 'pul':
            model_base = ElkanotoPuClassifier(SVC(probability=True))
            model_params_dist = dict(
                estimator=[SVC(probability=True), RandomForestClassifier()],
                hold_out_ratio=uniform(0.1, 0.2)
            )
        else:
            model_base = WeightedElkanotoPuClassifier(SVC(probability=True), labeled=10, unlabeled=20)
            model_params_dist = dict(
                estimator=[SVC(probability=True), RandomForestClassifier()],
                labeled=randint(5, 15),
                unlabeled=randint(5, 15),
                hold_out_ratio=uniform(0.1, 0.2)
            )
        self._logger.info(f'  model base {model_base}')
        self._logger.info(f'  model parameters list {model_params_dist}')
        return model_base, model_params_dist

    def _get_best_model(self):
        if self.model_name == 'ocsvm':
            model = OneClassSVM(
                kernel=self.best_params['kernel'],
                gamma=self.best_params['gamma'],
                nu=self.best_params['nu']
            )
        elif self.model_name == 'pul':
            model = ElkanotoPuClassifier(
                estimator=self.best_params['estimator'],
                hold_out_ratio=self.best_params['hold_out_ratio']
            )
        else:
            model = WeightedElkanotoPuClassifier(
                estimator=self.best_params['estimator'],
                labeled=self.best_params['labeled'],
                unlabeled=self.best_params['unlabeled'],
                hold_out_ratio=self.best_params['hold_out_ratio']
            )
        return model

    def predict_and_save(self, x, meta, cropland_mask=None):
        if cropland_mask is None:
            # predict from scratch
            to_name = self.to_name + '_from_scratch'
            y_preds = self._predict_from_scratch(x)
        else:
            # predict from cropland
            to_name = self.to_name + '_from_cropland'
            y_preds = self._predict_from_cropland(x, cropland_mask)
        self._save_predictions(meta, y_preds, to_name)

    def _predict_from_cropland(self, x, cropland_mask):
        self._logger.info('Predicting from cropland (with cropland only)...')
        x_cropland = x[cropland_mask]
        y_cropland_pred = self.model.predict(x_cropland)
        self._logger.info('Masking cropland region...')
        y_preds = np.zeros(x.shape[0], dtype=int)
        y_preds[cropland_mask] = y_cropland_pred
        self._logger.info('  ok')
        return y_preds

    def _predict_from_scratch(self, x):
        self._logger.info('Predicting from scratch (with all data)...')
        y_preds = self.model.predict(x)
        self._logger.info('  ok')
        return y_preds

    def _predict_pretrained_base_model(self, x):
        self._logger.info('Predicting x ...')
        preds = self.base_model.predict(x)
        return preds
