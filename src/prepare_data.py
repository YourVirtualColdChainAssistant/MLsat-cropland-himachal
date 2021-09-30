import pyproj
import datetime
import pandas as pd
import geopandas as gpd
from visualization import plot_timestamps
from util import load_target_shp, compute_mask, dropna_in_shapefile, stack_all_timestamps
from feature_engineering import add_bands, get_raw_monthly, get_statistics, get_difference


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
        plot_timestamps(self.timestamps_raw, '../figs/timestamps_raw.png')
        plot_timestamps(self.timestamps_weekly_ref, '../figs/timestamps_weekly.png')

        # 1. feature engineering
        df = self.feature_engineering(new_bands_name=['ndvi'])
        # 2. prepare labels
        self.merge_shapefiles()
        self.logger.info('Merged all shapefiles!')
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

    @staticmethod
    def merge_shapefiles(to_label_path='../data/all-labels/all-labels.shp'):
        # read all the shape files
        old_apples_shp = gpd.read_file('../data/apples/survey20210716_polygons20210819_corrected20210831.shp')
        new_apples_shp = gpd.read_file('../data/apples/survey20210825_polygons20210901_revised20210927.shp')
        non_crops_shp = gpd.read_file('../data/non-crops/non-crop.shp')
        other_crops_shp = gpd.read_file('../data/other-crops/other-crops.shp')
        # put all shape files into one geo dataframe
        all_labels_shp = gpd.GeoDataFrame(
            pd.concat([old_apples_shp, new_apples_shp, other_crops_shp, non_crops_shp], axis=0))
        all_labels_shp = all_labels_shp.dropna().reset_index(drop=True)  # delete empty polygons
        # mask for the study area
        study_area_shp = gpd.read_file('../data/study-area/study_area.shp')
        labels_in_study = gpd.overlay(all_labels_shp, study_area_shp, how='intersection')
        cols2drop = [col for col in ['id', 'id_2'] if col in labels_in_study.columns]
        labels_in_study = labels_in_study.drop(cols2drop, axis=1).rename(columns={'id_1': 'id'})
        labels_in_study.to_file(to_label_path)  # save to folder
