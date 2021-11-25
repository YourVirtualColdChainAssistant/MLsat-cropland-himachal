import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.utils.util import save_predictions_geotiff


class BaseModel(object):
    def __init__(self, logger, log_time, model_name, random_state):
        self._logger = logger
        self._log_time = log_time
        self.model_name = model_name
        self.random_state = random_state
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
        self.best_params = model_search.best_params_
        self._logger.info(f"  Best score {model_search.best_score_:.4f} with best parameters: {self.best_params}")

    def fit_and_save_best_model(self, x_train_val, y_train_val, best_params=None):
        # determin the best parameters
        if best_params is not None:
            self.best_params = best_params
        if self.best_params is None:
            raise ValueError('No best parameters are specified.')

        # fit model
        self._logger.info(f"Fitting the best {self.model_name.upper()} with {self.best_params}...")
        self.model = self._get_best_model()
        if self.model_name == 'ocsvm':
            self.model.fit(x_train_val)
        else:
            self.model.fit(x_train_val, y_train_val)

        # score training
        self._logger.info(f'  Fitted accuracy: {self.model.score(x_train_val, y_train_val):.4f}')

        # save model
        model_name = f'../models/{self.to_name}.pkl'
        pickle.dump(self.model, open(model_name, 'wb'))
        self._logger.info(f'  Saved the best {self.model_name.upper()} to {model_name}')

    @staticmethod
    def convert_partial_predictions(preds, index, meta):
        y_preds = np.zeros(meta['height'] * meta['width'])
        y_preds[index] = preds
        return y_preds

    def load_pretrained_model(self, pretrained_name):
        self._logger.info(f"Loading pretrained {pretrained_name}...")
        self.model = pickle.load(open(pretrained_name, 'rb'))
        self._logger.info('  ok')

    def _save_predictions(self, meta, y_preds, to_name=None):
        if to_name is None:
            to_name = self.to_name
        pred_path = f'../preds/{to_name}.tiff'
        save_predictions_geotiff(meta, y_preds, pred_path)
        self._logger.info(f'Saved {self.model_name.upper()} predictions to {pred_path}')

    def _get_model_base_and_params_list_grid(self, testing):
        raise NotImplementedError

    def _get_model_base_and_params_list_random(self):
        raise NotImplementedError

    def _get_best_model(self):
        raise NotImplementedError
