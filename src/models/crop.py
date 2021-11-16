import pickle
import numpy as np
from scipy.stats import uniform, randint
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from pulearn import ElkanotoPuClassifier, WeightedElkanotoPuClassifier
from src.models.base_model import BaseModel


class CropModel(BaseModel):
    def __init__(self, logger, log_time, model_name, random_state, pretrained_name=None):
        # inherent from parent
        super().__init__(logger, log_time, model_name, random_state)
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
