import pickle
import numpy as np
import pandas as pd


def get_cropland_mask(df, n_feature, pretrained_name):
    # read pretrained model
    trained_model = pickle.load(open(f'../models/{pretrained_name}.sav', 'rb'))

    # new columns
    df['gt_cropland'] = 0
    df.loc[(df.label.values == 1) | (df.label.values == 2), 'gt_cropland'] = 1

    # cropland
    to_pred_mask = df.label.values == 0
    preds = trained_model.predict(df.iloc[to_pred_mask, :n_feature].values)
    cropland_pred = np.empty_like(preds)
    cropland_pred[preds == 2] = 1
    cropland_pred[preds == 3] = 0
    df['gp_cropland'] = df['gt_cropland'].copy()
    df.loc[to_pred_mask, 'gp_cropland'] = cropland_pred
    df['gp_cropland_mask'] = False
    df.loc[df.gp_cropland.values == 1, 'gp_cropland_mask'] = True

    return df.gp_cropland_mask.values


def correct_predictions():
    pass


def resample_negatives(pos, neg):
    pass
