import pandas as pd
from skimage.draw import draw
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from util import *
from prepare_labels import *
from preprocessing import preprocess
from visualization import plot_timestamps, NVDI_profile


def main(args):
    # labels shapefile
    # merge_shapefiles()

    # check NDVI profile
    ndvi_profile = NVDI_profile(args.images_dir + 'clip/', '../data/all-labels/all-labels.shp')
    ndvi_profile.raw_profile()
    ndvi_profile.weekly_profile()
    ndvi_profile.monthly_profile()

    # stack timestamps
    bands_array, meta_train, timestamps_raw, timestamps_weekly, timestamps_weekly_ref = \
        stack_all_timestamps(args.images_dir + 'clip/')

    # plot timestamps
    plot_timestamps(timestamps_raw, '../figs/timestamps_raw.png')
    plot_timestamps(timestamps_weekly, '../figs/timestamps_weekly.png')

    # load shapefile labels
    print('*** Loading target shape files ***')
    # study region shapefile
    # _, study_rc_polygons, study_class_list = \
    #     load_target_shp('../data/study-area/study_area.shp',
    #                     transform=meta_train['transform'],
    #                     proj_out=pyproj.Proj(meta_train['crs']))
    # region_mask = compute_mask(study_rc_polygons, meta_train, study_class_list)
    # label shapefile
    train_polygons, train_rc_polygons, train_class_list = \
        load_target_shp('../data/all-labels/all-labels.shp',
                        transform=meta_train['transform'],
                        proj_out=pyproj.Proj(meta_train['crs']))
    train_mask = compute_mask(train_rc_polygons, meta_train, train_class_list)

    # feature engineering
    df = preprocess(timestamps_weekly_ref, bands_array, train_mask.reshape(-1), new_features=['ndvi'])
    # pairing x and y
    df['label'] = train_mask.reshape(-1)

    # select those with labels (deduct whose labels are 0)
    df_train_val = df[df['label'] != 0].reset_index(drop=True)
    # prepare x and y
    x = df_train_val.iloc[:, :-1].values
    y_3class = df_train_val.label.values  # raw data with 3 classes
    y = y_3class.copy()

    # modify to 2 classes
    y[y == 2] = 1
    print('With 3 classes:')
    count_classes(y_3class)
    print('With 2 classes:')
    count_classes(y)

    # split train validation set
    x_train, x_val, y_train, y_val = \
        train_test_split(x, y, test_size=0.2, random_state=42)
    print(f'features: {df.columns[:-1]}')
    print(f'x.shape {x.shape}, y.shape {y.shape}')
    print(f'x_train.shape {x_train.shape}, y_train.shape {y_train.shape}')
    print(f'x_val.shape {x_val.shape}, y_val.shape {y_val.shape}')

    ### models
    # SVM
    print("Training SVM...")
    svm = SVC()
    svm.fit(x_train, y_train)
    svm_score = svm.score(x_val, y_val)
    print(f'SVM accuracy {svm_score}')
    # save the model to disk
    pickle.dump(svm, open('../models/svm.sav', 'wb'))
    # load:: svm = pickle.load(open(filename, 'rb'))

    # Random forest
    print("Training RF...")
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    rfc_score = rfc.score(x_val, y_val)
    print(f'RF accuracy {rfc_score}')
    pickle.dump(svm, open('../models/rfc.sav', 'wb'))
    feature_importance_table(df_train_val.columns[:-1], rfc.feature_importances_, '../preds/rfc_importance.csv')


def merge_shapefiles(to_label_path='../data/all-labels/all-labels.shp'):
    # read all the shape files
    old_apples_shp = gpd.read_file('../data/apples/survey20210716_polygons20210819_corrected20210831.shp')
    new_apples_shp = gpd.read_file('../data/apples/survey20210825_polygons20210901.shp')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images_dir',
        type=str,
        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/')
    args = parser.parse_args()
    main(args)
