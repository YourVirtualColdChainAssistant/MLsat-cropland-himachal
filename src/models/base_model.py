import pickle
import numpy as np


def load_pretrained_model(self, pretrained_name):
    self._logger.info(f"Loading pretrained {pretrained_name}...")
    self.model = pickle.load(open(pretrained_name, 'rb'))
    self._logger.info('  ok')



def convert_partial_predictions(preds, index, meta):
    y_preds = np.zeros(meta['height'] * meta['width'])
    y_preds[index] = preds
    return y_preds
