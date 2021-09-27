import numpy as np
import pandas as pd


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
        print('No band is added.')
        return img
    else:
        print(f'Adding new features {new_features}')
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

        return np.append(img, np.stack(new_bands, axis=1), axis=1)


def get_raw_features(bands_name, num_of_weeks, bands_array):
    df_raw = pd.DataFrame()
    for i in range(num_of_weeks):
        new_col_name = [n + '_' + str(i+1) for n in bands_name]
        df_raw[new_col_name] = bands_array[:, :, i]  # fragmented df, please use pd.concat()
    return df_raw


def compute_statistics(op, data):
    if op == 'avg':
        return data.mean(axis=1)
    elif op == 'std':
        return data.std(axis=1)
    elif op == 'max':
        return data.max(axis=1)
    else:
        return 'No corresponding calculation.'


def get_statistics_by_band(band, num_of_weeks, df):
    cols = [band + '_' + str(i + 1) for i in range(num_of_weeks)]
    df_new = pd.DataFrame()
    for op in ['avg', 'std', 'max']:
        col_name = band + '_' + op
        df_new[col_name] = compute_statistics(op, df[cols])
    return df_new


def get_statistics(bands, num_of_weeks, df):
    print("Adding statistics as features...")
    df_new_list = []
    for band in bands:
        df_new_list.append(get_statistics_by_band(band, num_of_weeks, df))
    df_new = pd.concat(df_new_list, axis=1)
    return df_new


def get_difference_by_band(band, num_of_weeks, df):
    df_new = pd.DataFrame()
    for i in range(1, num_of_weeks):
        df_new[band + '_diff_' + str(i)] = df[band + '_' + str(i + 1)] - df[band + '_' + str(i)]  # fragmented df, please use pd.concat(
    return df_new


def get_difference(bands, num_of_weeks, df):
    print("Adding difference as features...")
    df_new_list = []
    for band in bands:
        df_new_list.append(get_difference_by_band(band, num_of_weeks, df))
    df_new = pd.concat(df_new_list, axis=1)
    return df_new


def preprocessing(num_of_weeks, bands_array, new_features=None):
    """
    Preprocessing includes
    :param num_of_weeks: int
    :param bands_array: array
        shape (height * width, num_of_ands, num_of_week)
    :param new_features: list of string
    :return:
    """
    print('Start preprocessing features...')
    bands_name = ['blue', 'green', 'red', 'nir']
    # add more features
    if new_features is not None:
        bands_array = add_features(bands_array, new_features)
        bands_name += new_features
    # raw features
    df = get_raw_features(bands_name, num_of_weeks, bands_array)
    df_list = [df]
    # statistics
    df_list.append(get_statistics(bands_name, num_of_weeks, df))
    # difference of two successive timestamps
    df_list.append(get_difference(bands_name, num_of_weeks, df))
    # concatenate
    df = pd.concat(df_list, axis=1)
    return df
