import numpy as np
import pandas as pd
from visualization import plot_ndvi_profile


def calculate_ndvi(red, nir):
    """ Compute the NDVI
        INPUT : red (np.array) -> the Red band images as a numpy array of float
                nir (np.array) -> the Near Infrared images as a numpy array of float
        OUTPUT : ndvi (np.array) -> the NDVI
    """
    ndvi = (nir - red) / (nir + red + 1e-12)
    return ndvi


def calculate_ndre(red_edge, nir):
    ndre = (nir - red_edge) / (nir + red_edge + 1e-12)
    return ndre


def calculate_gndvi(green, nir):
    gndvi = (nir - green) / (nir + green + 1e-12)
    return gndvi


def calculate_evi(red, blue, nir):
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    return evi


def calculate_cvi(green, red, nir):
    cvi = nir * red / (green + 1e-12)**2
    return cvi


def add_features(img, new_features=['ndvi']):
    """
    Add new features to the original bands.

    band02 = blue --> idx = 0
    band03 = green --> idx = 1
    band04 = red --> idx = 2
    band08 = nir --> idx = 3
    """
    if new_features is None:
        print(' No band is added.')
        return img
    else:
        print(f' Adding new features {new_features}...')
        new_bands = []

        # bands
        blue = img[:, 0]
        green = img[:, 1]
        red = img[:, 2]
        nir = img[:, 3]

        # add feature
        for feature in new_features:
            if feature == 'ndvi':
                new_bands.append(calculate_ndvi(red, nir))
            elif feature == 'gndvi':
                new_bands.append(calculate_gndvi(green, nir))
            elif feature == 'evi':
                new_bands.append(calculate_evi(blue, blue, nir))
            elif feature == 'cvi':
                new_bands.append(calculate_cvi(green, red, nir))
        print(f' Added {new_features}')
        return np.append(img, np.stack(new_bands, axis=1), axis=1)


def get_month_raw(bands_name, num_of_weeks, bands_array):
    print(f' Adding raw features...')
    df_new = pd.DataFrame()
    for i in np.arange(0, num_of_weeks, 4):
        new_col_name = [n + '_' + str(i+1) for n in bands_name]
        df_new[new_col_name] = bands_array[:, :, i]  # fragmented df, please use pd.concat()
    print(f' Added {df_new.shape[1]} new features.')
    return df_new.copy()


def compute_statistics(op, data):
    if op == 'avg':
        return data.mean(axis=1)
    elif op == 'std':
        return data.std(axis=1)
    elif op == 'max':
        return data.max(axis=1)
    else:
        return 'No corresponding calculation.'


def get_statistics_by_band(band, num_of_weeks, bands_array):
    cols = [band + '_' + str(i + 1) for i in range(num_of_weeks)]
    df_new = pd.DataFrame()
    for op in ['avg', 'std', 'max']:
        col_name = band + '_' + op
        df_new[col_name] = compute_statistics(op, bands_array)
    return df_new


def get_statistics(bands, num_of_weeks, bands_array):
    print(" Adding statistics...")
    df_new_list = []
    for i, band in enumerate(bands):
        df_new_list.append(get_statistics_by_band(band, num_of_weeks, bands_array[:, i, :]))
    df_new = pd.concat(df_new_list, axis=1)
    print(f' Added {df_new.shape[1]} new features.')
    return df_new.copy()


def get_difference_by_band(band, num_of_weeks, bands_array):
    df_new = pd.DataFrame()
    for i in range(1, num_of_weeks):
        df_new[band + '_diff_' + str(i)] = bands_array[:, i] - bands_array[:, i-1]
        # fragmented df, please use pd.concat()
    return df_new


def get_difference(bands, num_of_weeks, bands_array):
    print(" Adding difference...")
    df_new_list = []
    for i, band in enumerate(bands):
        df_new_list.append(get_difference_by_band(band, num_of_weeks, bands_array[:, i, :]))
    df_new = pd.concat(df_new_list, axis=1)
    print(f' Added {df_new.shape[1]} new features.')
    return df_new.copy()


def preprocess(timestamps_weekly, bands_array, train_mask, new_features=None):
    """
    Preprocessing includes
    :param timestamps_weekly: list
    :param bands_array: array
        shape (height * width, num_of_bands, num_of_week)
    :param train_mask: array
        shape
    :param new_features: list of string
    :return:
    """
    print('*** Feature engineering ***')
    bands_name = ['blue', 'green', 'red', 'nir']
    # add more features
    if new_features is not None:
        bands_array = add_features(bands_array, new_features)
        bands_name += new_features
    num_of_weeks = len(timestamps_weekly)

    # check ndvi profile
    print(' Plotting ndvi profile...')
    plot_ndvi_profile(bands_array[:, -1, :], train_mask, timestamps_weekly, '../figs/ndvi_profile.png')

    # raw features
    df = get_month_raw(bands_name, num_of_weeks, bands_array)
    df_list = [df]
    # statistics
    df_list.append(get_statistics(bands_name, num_of_weeks, bands_array))
    # difference of two successive timestamps
    df_list.append(get_difference(new_features, num_of_weeks, bands_array))
    # concatenate
    df = pd.concat(df_list, axis=1)
    print(f' Preprocess done! df.shape {df.shape}')
    return df
