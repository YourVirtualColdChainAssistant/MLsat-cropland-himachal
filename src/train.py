from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from .util import *
from .visualization import plot_feature_importance


def main(args):
    # labels shapefile
    dropna_in_shapefile('../data/labels/labels.shp', '../data/labels/labels_dropna.shp')

    # clip according to some shapefile
    # clip_all_raster(args.images_dir, shape_filepath='../data/study-area/study_area.shp')

    # stack the images of all timestamps
    stacked_raster = stack_all_timestamps(args.images_dir + 'clip/')
    stacked_band = np.array(stacked_raster['band'])
    meta_train = stacked_raster['meta']

    # load study area shapefile
    _, region_rc_polygons, region_class_list = \
        load_target_shp('../data/study-area/study_area.shp',
                        transform=meta_train['transform'],
                        proj_out=pyproj.Proj(meta_train['crs']))
    region_mask = compute_mask(region_rc_polygons, meta_train['width'], meta_train['height'], region_class_list)
    # load label shapefile
    train_polygons, train_rc_polygons, train_class_list = \
        load_target_shp('../data/labels/labels_dropna.shp',
                        transform=meta_train['transform'],
                        proj_out=pyproj.Proj(meta_train['crs']))
    train_mask = compute_mask(train_rc_polygons, meta_train['width'], meta_train['height'], train_class_list)

    # processing bands
    df = preprocessing(stacked_band)
    # pairing x and y
    df['label'] = train_mask.reshape(-1)
    # select those with class labels
    df_train_val = df[df['label'] != 0].reset_index(drop=True)
    # prepare x and y
    x = df_train_val.iloc[:, :-1].values
    y = df_train_val.label.values
    print(f'x.shape {x.shape}, y.shape {y.shape}')
    count_classes(y)
    # split train validation set
    x_train, x_val, y_train, y_val = \
        train_test_split(x, y, test_size=0.2, random_state=42)

    ### models
    # SVM
    svm = SVC()
    svm.fit(x_train, y_train)
    svm_score = svm.score(x_val, y_val)
    print(f'svm accuracy {svm_score}')
    # Random forest
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    rfc_score = rfc.score(x_val, y_val)
    print(f'rf accuracy {rfc_score}')
    plot_feature_importance(df_train_val.columns[:-1], rfc.feature_importances_)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images_dir',
        type=str,
        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/')
    args = parser.parse_args()
    main(args)