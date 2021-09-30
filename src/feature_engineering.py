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


def add_bands(logger, img, new_bands_name=['ndvi']):
    """
    Add new features to the original bands.

    band02 = blue --> idx = 0
    band03 = green --> idx = 1
    band04 = red --> idx = 2
    band08 = nir --> idx = 3
    """
    if new_bands_name is None:
        logger.info('  No band is added.')
        return img
    else:
        logger.info(f'  Adding new bands {new_bands_name}...')
        new_bands = []

        # bands
        blue = img[:, 0, :]
        green = img[:, 1, :]
        red = img[:, 2, :]
        nir = img[:, 3, :]

        # add feature
        for new_band_name in new_bands_name:
            if new_band_name == 'ndvi':
                new_bands.append(calculate_ndvi(red, nir))
            elif new_band_name == 'gndvi':
                new_bands.append(calculate_gndvi(green, nir))
            elif new_band_name == 'evi':
                new_bands.append(calculate_evi(blue, blue, nir))
            elif new_band_name == 'cvi':
                new_bands.append(calculate_cvi(green, red, nir))
        logger.info('  ok')

        return np.append(img, np.stack(new_bands, axis=1), axis=1)


def get_raw_monthly(logger, bands_name, num_of_weeks, bands_array):
    logger.info('  Adding raw features...')
    df_new = pd.DataFrame()
    for i in np.arange(0, num_of_weeks, 4):
        new_col_name = [n + '_' + str(i+1) for n in bands_name]
        df_new[new_col_name] = bands_array[:, :, i]  # fragmented df, please use pd.concat()
    logger.info(f'  ok, {df_new.shape[1]} new features are added.')
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


def get_statistics_by_band(band_name, num_of_weeks, bands_array):
    df_new = pd.DataFrame()
    for op in ['avg', 'std', 'max']:
        col_name = band_name + '_' + op
        df_new[col_name] = compute_statistics(op, bands_array)
    return df_new


def get_statistics(logger, bands_name, num_of_weeks, bands_array):
    logger.info("  Adding statistics...")
    df_new_list = []
    for i, band_name in enumerate(bands_name):
        df_new_list.append(get_statistics_by_band(band_name, num_of_weeks, bands_array[:, i, :]))
    df_new = pd.concat(df_new_list, axis=1)
    logger.info(f'  ok, {df_new.shape[1]} new features are added.')
    return df_new.copy()


def get_difference_by_band(band_name, num_of_weeks, bands_array):
    df_new = pd.DataFrame()
    for i in range(1, num_of_weeks):
        df_new[band_name + '_diff_' + str(i)] = bands_array[:, i] - bands_array[:, i-1]
        # fragmented df, please use pd.concat()
    return df_new


def get_difference(logger, bands_name, num_of_weeks, bands_array):
    logger.info("  Adding difference...")
    df_new_list = []
    for i, band_name in enumerate(bands_name):
        df_new_list.append(get_difference_by_band(band_name, num_of_weeks, bands_array[:, i, :]))
    df_new = pd.concat(df_new_list, axis=1)
    logger.info(f'  ok, {df_new.shape[1]} new features are added.')
    return df_new.copy()
