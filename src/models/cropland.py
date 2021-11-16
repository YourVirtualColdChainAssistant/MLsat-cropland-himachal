import pickle
from scipy.stats import uniform, loguniform, randint
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from src.evaluation.evaluate import align_raster, clip_open_datasets_based_on_shp_region, \
    compare_predictions_with_gfsad, compare_predictions_with_copernicus, \
    impurity_importance_table, permutation_importance_table
from src.models.base_model import BaseModel


class CroplandModel(BaseModel):
    def __init__(self, logger, log_time, model_name, random_state, pretrained_name=None):
        # inherent from parent
        super().__init__(logger, log_time, model_name, random_state)
        self._check_model_name(model_list=['svc', 'rfc', 'mlp'])
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
        clip_open_datasets_based_on_shp_region(gfsad_path, gfsad_clip_path)
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
        clip_open_datasets_based_on_shp_region(copernicus_path, copernicus_clip_path)
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
                    kernel=['poly', 'rbf'],
                    random_state=[self.random_state]
                )
            else:
                model_params_list = dict(
                    C=[1],
                    gamma=['scale'],
                    kernel=['rbf'],
                    random_state=[self.random_state]
                )
        elif self.model_name == 'rfc':
            model_base = RandomForestClassifier()
            if not testing:
                model_params_list = dict(
                    n_estimators=[100, 300, 500],
                    criterion=['gini', 'entropy'],
                    max_depth=[5, 10, 15],
                    max_samples=[0.5, 0.8, 1],
                    random_state=[self.random_state]
                )
            else:
                model_params_list = dict(
                    n_estimators=[100],
                    criterion=['gini'],
                    max_depth=[5],
                    max_samples=[0.8],
                    random_state=[self.random_state]
                )
        else:  # self.model_name == 'mlp':
            model_base = MLPClassifier()
            if not testing:
                model_params_list = dict(
                    hidden_layer_sizes=[(100,), (300,), (300, 300)],
                    alpha=[0.0001, 0.0005, 0.001, 0.005],
                    max_iter=[200, 500],
                    activation=['relu'],
                    early_stopping=[True],
                    random_state=[self.random_state]
                )
            else:
                model_params_list = dict(
                    hidden_layer_sizes=[(100,)],
                    alpha=[0.0001],
                    max_iter=[200],
                    activation=['relu'],
                    early_stopping=[True],
                    random_state=[self.random_state]
                )
        self._logger.info(f'  model base {model_base}')
        self._logger.info(f'  model parameters list {model_params_list}')
        return model_base, model_params_list

    def _get_model_base_and_params_list_random(self):
        if self.model_name == 'svc':
            model_base = SVC()
            model_params_dist = dict(
                C=loguniform(0.1, 100),
                gamma=['scale', 'auto'],
                kernel=['poly', 'rbf'],
                random_state=[self.random_state]
            )
        elif self.model_name == 'rfc':
            model_base = RandomForestClassifier()
            model_params_dist = dict(
                n_estimators=randint(100, 1000),
                criterion=['gini', 'entropy'],
                max_depth=randint(5, 15),
                max_samples=uniform(0.5, 0.5),  # uniform(loc, scale) -> a=loc, b=loc+scale
                random_state=[self.random_state]
            )
        else:  # self.model_name == 'mlp':
            model_base = MLPClassifier()
            model_params_dist = dict(
                hidden_layer_sizes=[(100,), (300,)],
                alpha=loguniform(0.0001, 0.001),
                max_iter=randint(200, 500),
                activation=['relu'],
                early_stopping=[True],  # uniform(loc, scale) -> a=loc, b=loc+scale
                random_state=[self.random_state]
            )
        self._logger.info(f'  model base {model_base}')
        self._logger.info(f'  model parameters distribution {model_params_dist}')
        return model_base, model_params_dist

    def _get_best_model(self):
        if self.model_name == 'svc':
            model = SVC(
                C=self.best_params['C'],
                gamma=self.best_params['gamma'],
                kernel=self.best_params['kernel'],
                random_state=self.random_state
            )
        elif self.model_name == 'rfc':
            model = RandomForestClassifier(
                n_estimators=self.best_params['n_estimators'],
                criterion=self.best_params['criterion'],
                max_depth=self.best_params['max_depth'],
                max_samples=self.best_params['max_samples'],
                random_state=self.random_state
            )
        else:  # self.model_name == 'mlp':
            model = MLPClassifier(
                hidden_layer_sizes=self.best_params['hidden_layer_sizes'],
                alpha=self.best_params['alpha'],
                max_iter=self.best_params['max_iter'],
                activation=self.best_params['activation'],
                early_stopping=self.best_params['early_stopping'],
                random_state=self.random_state
            )
        return model
