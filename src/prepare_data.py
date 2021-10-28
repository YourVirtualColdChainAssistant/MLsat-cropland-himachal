import pyproj
import datetime
import math
import random
import fiona
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import skimage
import skimage.draw
import pyproj
from visualization import plot_timestamps
from util import stack_all_timestamps, count_classes, get_grid_idx, load_target_shp, compute_mask
from feature_engineering import add_bands, get_raw_monthly, get_statistics, get_difference
from sklearn.model_selection import train_test_split as random_train_test_split
from sklearn.model_selection import GroupKFold


class Pipeline(object):
    def __init__(self, logger, from_dir, way='weekly', interpolation='previous'):
        self.logger = logger
        self.bands_array, self.meta, self.timestamps_raw, self.timestamps_weekly, self.timestamps_weekly_ref = \
            stack_all_timestamps(self.logger, from_dir, way=way, interpolation=interpolation)

    def pipeline(self):
        """
        To prepare data for training and testing
        :return:
        """
        # timestamps visualization
        # plot_timestamps(self.timestamps_raw, '../figs/timestamps_raw.png')
        # plot_timestamps(self.timestamps_weekly_ref, '../figs/timestamps_weekly.png')

        # 1. feature engineering
        df = self.feature_engineering(new_bands_name=['ndvi'])
        # 2. prepare labels
        train_mask = self.label_preparation()
        # 3. pair features and labels
        df['label'] = train_mask.reshape(-1)
        return df

    def feature_engineering(self, new_bands_name=['ndvi']):
        self.logger.info('--- Features engineering ---')
        bands_name = ['blue', 'green', 'red', 'nir']
        # add more features
        if new_bands_name is not None:
            self.bands_array = add_bands(self.logger, self.bands_array, new_bands_name)
            bands_name += new_bands_name
        num_of_weeks = len(self.timestamps_weekly_ref)

        # raw features
        df = get_raw_monthly(self.logger, bands_name, num_of_weeks, self.bands_array)
        df_list = [df]
        # statistics
        df_list.append(get_statistics(self.logger, bands_name, num_of_weeks, self.bands_array))
        # difference of two successive timestamps
        df_list.append(get_difference(self.logger, new_bands_name, num_of_weeks, self.bands_array))
        # concatenate
        df = pd.concat(df_list, axis=1)
        self.logger.info(f'Done! df.shape={df.shape}')
        return df

    def label_preparation(self):
        self.logger.info('--- Load target shape files ---')
        # study region shapefile
        # _, study_rc_polygons, study_class_list = \
        #     load_target_shp('../data/study-area/study_area.shp',
        #                     transform=meta_train['transform'],
        #                     proj_out=pyproj.Proj(meta_train['crs']))
        # region_mask = compute_mask(study_rc_polygons, meta_train, study_class_list)
        # label shapefile
        train_polygons, train_rc_polygons, train_class_list = \
            load_target_shp('../data/all-labels/all-labels.shp',
                            transform=self.meta['transform'],
                            proj_out=pyproj.Proj(self.meta['crs']))
        train_mask = compute_mask(train_rc_polygons, self.meta, train_class_list)
        self.logger.info(' Done!')
        return train_mask


def train_test_split(logger, df, train_val_test_mask, num_feature, binary,
                     split_by='random', test_ratio=0.2, random_seed=42):
    """

    :param logger:
    :param df:
    :param split_by:
    :param train_val_test_mask: bool pd.Series
    :param num_feature: int
    :param binary: bool
    :param test_ratio: float
    :param random_seed: int

    :return:
    x_train_val: np.array
        shape (num_train_val, num_features)
    x_test: np.array
        shape (num_test, num_features)
    y_train_val: np.array, int
        shape (num_train_val, )
    y_test: np.array, int
        shape (num_test, )
    grid_idx_train_val: np.array, int
        shape (num_train_val, )
    grid_idx_test: np.array, int
        shape (num_test, )
    """
    random.seed(random_seed)
    # data summary
    logger.info('--- Print data summary ---')
    # 3 classes
    logger.info('y_train_val_test with 3 classes:')
    count_classes(logger, df[df['label'] != 0].label.values)
    if binary:
        # modify to 2 classes
        df.loc[df.label.values == 1, 'label'] = 2
        logger.info('y_train_val_test with 2 classes:')
        count_classes(logger, df[df['label'] != 0].label.values)

    # start to split
    if split_by == 'random':
        df_train_val_test = df[train_val_test_mask].reset_index(drop=True)
        x_train_val_test = df_train_val_test.iloc[:, :num_feature].values
        y_train_val_test = df_train_val_test.label.values
        x_train_val, x_test, y_train_val, y_test = \
            random_train_test_split(x_train_val_test, y_train_val_test,
                                    test_size=test_ratio, random_state=random_seed)
        grid_idx_train_val, grid_idx_test = None, None
    else:
        # split_by == 'spatial':
        # train, validation, test split
        df_train_val_test = df[train_val_test_mask].reset_index(drop=True)
        # ordered
        unique_grid_idx_labeled = list(np.unique(df_train_val_test.grid_idx.values)).sort()
        # might be random
        unique_grid_idx_test = unique_grid_idx_labeled[:math.ceil(len(unique_grid_idx_labeled) * test_ratio)]
        # unique_grid_idx_test = random.sample(unique_grid_idx_labeled,
        #                                      math.ceil(len(unique_grid_idx_labeled) * test_ratio))
        unique_grid_idx_train_val = list(set(unique_grid_idx_labeled) - set(unique_grid_idx_test))
        # mask
        train_val_mask = [True if idx in unique_grid_idx_train_val else False for idx in
                          df_train_val_test.grid_idx.values]
        test_mask = [True if idx in unique_grid_idx_test else False for idx in df_train_val_test.grid_idx.values]
        # get corresponding data
        x_train_val = df_train_val_test.iloc[train_val_mask, :num_feature].values
        y_train_val = df_train_val_test.loc[train_val_mask, 'label'].values
        grid_idx_train_val = df_train_val_test.loc[train_val_mask, 'grid_idx'].values
        x_test = df_train_val_test.iloc[test_mask, :num_feature].values
        y_test = df_train_val_test.loc[test_mask, 'label'].values
        grid_idx_test = df_train_val_test.loc[test_mask, 'grid_idx'].values
    logger.info('y_train_val:')
    count_classes(logger, y_train_val)
    logger.info('y_test:')
    count_classes(logger, y_test)
    return x_train_val, x_test, y_train_val, y_test, grid_idx_train_val, grid_idx_test


def get_spatial_cv_fold(x_train_val, y_train_val, grid_idx_train_val):
    grid_kfold = GroupKFold(n_splits=4)
    # generator for the train/test indices
    data_kfold = grid_kfold.split(x_train_val, y_train_val, grid_idx_train_val)
    # create a nested list of train and test indices for each fold
    train_indices, val_indices = [list(train_val) for train_val in zip(*data_kfold)]
    data_cv = [*zip(train_indices, val_indices)]
    grid_idx_fold = [grid_idx_train_val[val_indice] for val_indice in val_indices]
    return data_cv, grid_idx_fold


def get_unlabeled_data(df, unlabeled_mask, num_feature):
    x_unl = df.iloc[unlabeled_mask, :num_feature].values
    y_unl = np.zeros(x_unl.shape[0], dtype=int)
    grid_idx_unl = df.loc[unlabeled_mask, 'grid_idx'].values
    return x_unl, y_unl, grid_idx_unl
