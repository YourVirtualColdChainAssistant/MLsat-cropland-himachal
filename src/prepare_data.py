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


def train_test_split(logger, df, by, spatial_dict=None, test_ratio=0.2):
    """

    :param logger:
    :param df:
    :param by:
    :param spatial_dict:
        {
        cell_size: int,
        height: int,
        width: int,
        }
    :param test_ratio: float
    :return:
    """
    random_seed = 22
    random.seed(random_seed)
    # data summary
    logger.info('--- Print data summary ---')
    # 3 classes
    logger.info('With 3 classes:')
    count_classes(logger, df[df['label'] != 0].label.values)
    # modify to 2 classes
    df.loc[df.label.values == 2, 'label'] = 1
    logger.info('With 2 classes:')
    count_classes(logger, df[df['label'] != 0].label.values)

    # start to split
    if by == 'random':
        df_train_val_test = df[df['label'] != 0].reset_index(drop=True)
        x_train_val_test = df_train_val_test.iloc[:, :-1].values
        y_train_val_test = df_train_val_test.label.values
        x_train_val, x_test, y_train_val, y_test = \
            random_train_test_split(x_train_val_test, y_train_val_test,
                                    test_size=test_ratio, random_state=random_seed)
        grid_idx_train_val, grid_idx_test = None, None
    elif by == 'spatial':
        if spatial_dict is None:
            logger.info('Please assign spatial_dict.')
        else:
            cell_size, height, width = spatial_dict['cell_size'], spatial_dict['height'], spatial_dict['width']
            grid_idx = list(get_grid_idx(cell_size, height, width).reshape(-1))
            # match grid idx with df
            df['grid_idx'] = grid_idx
            unique_grid_idx_labeled = list(df.loc[df['label'] == 1, 'grid_idx'].unique())
            unique_grid_idx_test = random.sample(unique_grid_idx_labeled,
                                                 math.ceil(unique_grid_idx_labeled.shape[0] * test_ratio))
            unique_grid_idx_train_val = list(set(unique_grid_idx_labeled) - set(unique_grid_idx_test))

            # train, validation, test split
            df_train_val_test = df[df['label'] != 0].reset_index(drop=True)
            train_val_mask = [True if idx in unique_grid_idx_train_val else False for idx in
                              df_train_val_test.grid_idx.values]
            test_mask = [True if idx in unique_grid_idx_test else False for idx in df_train_val_test.grid_idx.values]
            x_train_val = df_train_val_test.loc[train_val_mask, :-2]
            y_train_val = df_train_val_test.loc[train_val_mask, 'label']
            grid_idx_train_val = df_train_val_test.loc[train_val_mask, 'grid_idx']
            x_test = df_train_val_test.loc[test_mask, :-2]
            y_test = df_train_val_test.loc[test_mask, 'label']
            grid_idx_test = df_train_val_test.loc[test_mask, 'grid_idx']
    else:
        logger.info('Please choose from by=[random, spatial].')
        exit()
    return x_train_val, x_test, y_train_val, y_test, grid_idx_train_val, grid_idx_test


def spatial_cross_validation(x_train_val, y_train_val, grid_idx_train_val):
    grid_kfold = GroupKFold(n_splits=4)
    # generator for the train/test indices
    data_kfold = grid_kfold.split(x_train_val, y_train_val, grid_idx_train_val)
    # create a nested list of train and test indices for each fold
    train_indices, val_indices = [list(train_val) for train_val in zip(*grid_kfold)]
    data_cv = [*zip(train_indices, val_indices)]
    return data_cv
