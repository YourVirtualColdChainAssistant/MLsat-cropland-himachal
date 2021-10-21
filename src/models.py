import pickle
from scipy.stats import uniform, loguniform, randint
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from util import save_predictions_geotiff
from evaluation import align_raster, clip_open_datasets_based_on_study_area, \
    compare_predictions_with_gfsad, compare_predictions_with_copernicus, \
    impurity_importance_table, permutation_importance_table


class Model(object):
    def __init__(self, logger, log_time, model_name):
        self.logger = logger
        self.log_time = log_time
        self.model_name = model_name
        model_list = ['svc', 'rfc', 'mlp', 'gru']
        if self.model_name not in model_list:
            self.logger.info(f'No such model {self.model_name}. Please choose from {model_list}.')
        self.logger.info(f'===== {self.model_name.upper()} =====')
        self.best_params = None
        self.model = None

    def find_best_parameters(self, x_train_val, y_train_val, by='grid', cv=4):
        self.logger.info(f"Finding the best parameters using {by} search...")
        if by == 'grid':
            model_base, model_params = self._get_model_base_and_params_list_grid()
            model_search = GridSearchCV(estimator=model_base, param_grid=model_params,
                                        cv=cv, verbose=3, n_jobs=-1)
        elif by == 'random':
            model_base, model_params = self._get_model_base_and_params_list_random()
            model_search = RandomizedSearchCV(estimator=model_base, param_distributions=model_params,
                                              cv=cv, verbose=3, n_jobs=-1, n_iter=1)
        else:
            self.logger.info(f'No {by} search. Choose from ["grid", "random"]')
            exit()

        model_search.fit(x_train_val, y_train_val)
        self.logger.info('  ok')
        self.logger.info(f"\n{model_search.cv_results_}")
        model_search_name = f'../models/{self.log_time}_{self.model_name}_{by}.sav'
        pickle.dump(model_search, open(model_search_name, 'wb'))
        self.logger.info(f'  Saved {self.model_name.upper()} {by} search model to {model_search_name}')
        self.best_params = model_search.best_params_
        self.logger.info(f"  Best score {model_search.best_score_:.4f} with best parameters: {self.best_params}")

    def fit_and_save_best_model(self, x_train_val, y_train_val, best_params=None):
        if best_params is not None:
            self.best_params = best_params
        self.logger.info(f"Fitting the best {self.model_name.upper()} with {self.best_params}...")
        self.model = self._get_best_model()
        self.model.fit(x_train_val, y_train_val)
        self.logger.info('  ok')
        model_name = f'../models/{self.log_time}_{self.model_name}.sav'
        pickle.dump(self.model, open(model_name, 'wb'))
        self.logger.info(f'  Saved the best {self.model_name.upper()} to {model_name}')

    def save_predictions(self, x, meta):
        self.logger.info('Predicting all the data...')
        y_preds = self.model.predict(x)
        self.logger.info('  ok')
        preds_name = f'../preds/{self.log_time}_{self.model_name}.tif'
        save_predictions_geotiff(meta, y_preds, preds_name)
        self.logger.info(f'  Saved {self.model_name.upper()} predictions to {preds_name}')

    def evaluate_all(self, x_test, y_test, feature_names):
        self.logger.info('Evaluating by metrics...')
        self.evaluate_by_metrics(x_test, y_test)
        self.logger.info('Evaluating by feature importance...')
        self.evaluate_by_feature_importance(x_test, y_test, feature_names)
        self.logger.info('Evaluating by open datasets...')
        self.evaluate_by_open_datasets()

    def evaluate_by_metrics(self, x_test, y_test):
        self.logger.info("Predicting test data...")
        y_test_pred = self.model.predict(x_test)
        self.logger.info('  ok')
        self.logger.info(
            f"\n{classification_report(y_test, y_test_pred, labels=[1, 3], target_names=['crops', 'non-crops'])}"
        )

    def evaluate_by_feature_importance(self, x_test, y_test, feature_names):
        model_PI = f'../preds/{self.log_time}_{self.model_name}_PI.csv'
        permutation_importance_table(self.model, x_test, y_test, feature_names, f'{model_PI}')
        self.logger.info('  ok')
        self.logger.info(f'  Saved permutation importance to {model_PI}')
        if self.model_name == 'rfc':
            model_II = f'../preds/{self.log_time}_{self.model_name}_II.csv'
            impurity_importance_table(feature_names, self.model.feature_importances_, f'{model_II}')
            self.logger.info(f'  Saved impurity importance to {model_II}')

    def evaluate_by_open_datasets(self, pred_path=None):
        if pred_path is None:
            pred_path = f'../preds/{self.log_time}_{self.model_name}.tif'
        ancilliary_path = 'K:/2021-data-org/4. RESEARCH_n/ML/MLsatellite/Data/layers_india/ancilliary_data/'

        # gfsad
        gfsad_path = ancilliary_path + 'cropland/GFSAD30/GFSAD30SAAFGIRCE_2015_N30E70_001_2017286103800.tif'
        gfsad_clip_path = '../data/gfsad_clipped.tiff'
        gfsad_align_path = '../data/gfsad_aligned.tiff'

        self.logger.info('Comparing GFSAD dataset with predictions...')
        clip_open_datasets_based_on_study_area(gfsad_path, gfsad_clip_path)
        self.logger.info('  Clipped GFSAD dataset to study_region.')
        align_raster(pred_path, gfsad_clip_path, gfsad_align_path)
        self.logger.info('  Aligned GFSAD dataset to predictions.')
        compare_predictions_with_gfsad(pred_path, gfsad_align_path, self.logger)

        # copernicus
        copernicus_path = ancilliary_path + 'landcover/Copernicus_LC100m/INDIA_2019/' + \
                          'E060N40_PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif'
        copernicus_clip_path = '../data/copernicus_clipped.tiff'
        copernicus_align_path = '../data/copernicus_aligned.tiff'

        self.logger.info('Comparing Copernicus dataset with predictions...')
        clip_open_datasets_based_on_study_area(copernicus_path, copernicus_clip_path)
        self.logger.info('  Clipped Copernicus dataset to study_region.')
        align_raster(pred_path, copernicus_clip_path, copernicus_align_path)
        self.logger.info('  Aligned Copernicus dataset to predictions.')
        compare_predictions_with_copernicus(pred_path, copernicus_align_path, self.logger)

    def load_pretrained_model(self, pretrained_name):
        self.logger.info(f"Loading pretrained {pretrained_name}...")
        self.model = pickle.load(open(pretrained_name, 'rb'))
        self.logger.info('  ok')

    def _get_model_base_and_params_list_grid(self):
        if self.model_name == 'svc':
            model_base = SVC()
            # model_params_list = dict(
            #     C=[0.5, 1, 10, 100],
            #     kernel=['linear', 'poly', 'rbf']
            # )
            model_params_list = dict(
                C=[1],
                kernel=['linear']
            )
        elif self.model_name == 'rfc':
            model_base = RandomForestClassifier()
            # model_params_list = dict(
            #     n_estimators=[100, 300, 500],
            #     criterion=['gini', 'entropy'],
            #     max_depth=[5, 10, 15],
            #     max_samples=[0.5, 0.8, 1]
            # )
            model_params_list = dict(
                n_estimators=[100],
                criterion=['gini'],
                max_depth=[5],
                max_samples=[0.8]
            )
        elif self.model_name == 'mlp':
            model_base = MLPClassifier()
            # model_params_list = dict(
            #     hidden_layer_sizes = [(100,), (300,), (300,300)],
            #     alpha = [0.0001, 0.0005, 0.001, 0.005],
            #     max_iter=[200, 500],
            #     activation=['relu'],
            #     early_stopping=[True]
            # )
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
        self.logger.info(f'  model base {model_base}')
        self.logger.info(f'  model parameters list {model_params_list}')
        return model_base, model_params_list

    def _get_model_base_and_params_list_random(self):
        if self.model_name == 'svc':
            model_base = SVC()
            model_params_dist = dict(
                C=loguniform(0.1, 100),
                kernel=['linear', 'poly', 'rbf']
            )
        elif self.model_name == 'rfc':
            model_base = RandomForestClassifier()
            model_params_dist = dict(
                n_estimators=randint(100, 1000),
                criterion=['gini', 'entropy'],
                max_depth=randint(5, 15),
                max_samples=uniform(0.5, 1)
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
        self.logger.info(f'  model base {model_base}')
        self.logger.info(f'  model parameters distribution {model_params_dist}')
        return model_base, model_params_dist

    def _get_best_model(self):
        if self.model_name == 'svc':
            model = SVC(
                C=self.best_params['C'],
                kernel=self.best_params['kernel'])
        elif self.model_name == 'rfc':
            model = RandomForestClassifier(
                n_estimators=self.best_params['n_estimators'],
                criterion=self.best_params['criterion'],
                max_depth=self.best_params['max_depth'],
                max_samples=self.best_params['max_samples'])
        elif self.model_name == 'mlp':
            model = MLPClassifier(
                hidden_layer_sizes=self.best_params['hidden_layer_sizes'],
                alpha=self.best_params['alpha'],
                max_iter=self.best_params['max_iter'],
                activation=self.best_params['activation'],
                early_stopping=self.best_params['early_stopping'])
        else:
            model = None
        return model
