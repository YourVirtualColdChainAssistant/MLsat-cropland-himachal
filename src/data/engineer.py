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


# TODO: vegetation indexes which utilize low-resolution bands
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
    cvi = nir * red / (green + 1e-12) ** 2
    return cvi


def add_bands(logger, img, descriptions, new_bands_name=['ndvi']):
    """
    Add new features to the original bands.

    band02 = blue --> idx = 0
    band03 = green --> idx = 1
    band04 = red --> idx = 2
    band08 = nir --> idx = 3

    Parameters
    ----------
    logger
    img: shape (height, width, n_bands, n_weeks)
    descriptions
    new_bands_name

    Returns
    -------

    """
    if new_bands_name is None:
        logger.info('No band is added.')
        return img
    else:
        logger.info(f'Adding new bands {new_bands_name}...')
        new_bands = []

        band_map = {
            'blue': 'B02',
            'green': 'B03',
            'red': 'B04',
            'nir': 'B08'
        }

        # bands
        blue = img[:, :, descriptions.index(band_map['blue']), :]
        green = img[:, :, descriptions.index(band_map['green']), :]
        red = img[:, :, descriptions.index(band_map['red']), :]
        nir = img[:, :, descriptions.index(band_map['nir']), :]

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

        return np.append(img, np.stack(new_bands, axis=2), axis=2)


def get_raw_every_n_weeks(logger, bands_name, n_weeks, bands_array, n=4):
    """

    Parameters
    ----------
    logger
    bands_name
    n_weeks
    bands_array: np.array
        shape (n_pixels, n_bands, n_weeks)
    n

    Returns
    -------

    """
    logger.info(f'Adding raw features every {n} weeks...')
    df_new = pd.DataFrame()
    for i in np.arange(0, n_weeks, n):
        new_col_name = [n + '_' + str(i + 1) for n in bands_name]
        if n_weeks - i >= n:
            df_new = pd.concat([df_new, pd.DataFrame(bands_array[:, :, i:i + n].max(axis=2), columns=new_col_name)],
                               axis=1)  # take max
        else:
            df_new = pd.concat([df_new, pd.DataFrame(bands_array[:, :, i:].max(axis=2), columns=new_col_name)],
                               axis=1)  # take max
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


def get_statistics_by_band(band_name, bands_array):
    df_new = pd.DataFrame()
    for op in ['avg', 'std', 'max']:
        col_name = band_name + '_' + op
        df_new = pd.concat([df_new, pd.DataFrame(compute_statistics(op, bands_array), columns=[col_name])], axis=1)
    return df_new


def get_statistics(logger, bands_name, bands_array):
    logger.info("Adding statistics...")
    df_new_list = []
    for i, band_name in enumerate(bands_name):
        df_new_list.append(get_statistics_by_band(band_name, bands_array[:, i, :]))
    df_new = pd.concat(df_new_list, axis=1)
    logger.info(f'  ok, {df_new.shape[1]} new features are added.')
    return df_new.copy()


def get_difference_by_band(band_name, n_weeks, bands_array):
    df_new = pd.DataFrame()
    for i in range(1, n_weeks):
        col_name = band_name + '_diff_' + str(i)
        df_new = pd.concat([df_new, pd.DataFrame(bands_array[:, i] - bands_array[:, i - 1], columns=[col_name])],
                           axis=1)
        # fragmented df, please use pd.concat()
    return df_new


def get_difference(logger, bands_name, n_weeks, bands_array):
    """

    Parameters
    ----------
    logger
    bands_name: list of string
        A list of new bands name.
    n_weeks: int
    bands_array: np.array
        shape (n_pixels, n_new_bands, n_weeks)

    Returns
    -------

    """
    logger.info("Adding difference...")
    df_new_list = []
    for i, band_name in enumerate(bands_name):
        df_new_list.append(get_difference_by_band(band_name, n_weeks, bands_array[:, i, :]))
    df_new = pd.concat(df_new_list, axis=1)
    logger.info(f'  ok, {df_new.shape[1]} new features are added.')
    return df_new.copy()


def get_spatial_features(logger, arr):
    """

    Parameters
    ----------
    logger
    arr：np.array of ndvi time series
        shape (height, width, n_weeks)

    Returns
    -------

    """
    logger.info('Adding spatial features of NDVI')
    height, width, n_weeks = arr.shape
    mean_list, std_list = [], []
    for r in range(height):
        for c in range(width):
            neighbors = get_neighbors(arr, r, c)
            mean_list.append(neighbors.mean(axis=(0, 1)))
            std_list.append(neighbors.std(axis=(0, 1)))
    df_mean = pd.DataFrame(mean_list, columns=['ndvi_spat_mean_' + str(w) for w in range(n_weeks)])
    df_std = pd.DataFrame(std_list, columns=['ndvi_spat_std_' + str(w) for w in range(n_weeks)])
    df = pd.concat([df_mean, df_std], axis=1)
    logger.info(f"  ok, {df.shape[1]} spatial features are added.")
    return df


def get_all_spatial_features(logger, bands_name, arr):
    logger.info('Adding all spatial features')
    height, width, _, n_weeks = arr.shape
    df = pd.DataFrame()
    for i, b in enumerate(bands_name): 
        mean_list, std_list = [], []
        for r in range(height):
            for c in range(width):
                neighbors = get_neighbors(arr[:, :, i, :], r, c)
                mean_list.append(neighbors.mean(axis=(0, 1)))
                std_list.append(neighbors.std(axis=(0, 1)))
        df_mean = pd.DataFrame(mean_list, columns=[b + '_spat_mean_' + str(w) for w in range(n_weeks)])
        df_std = pd.DataFrame(std_list, columns=[b + '_spat_std_' + str(w) for w in range(n_weeks)])
        df = pd.concat([df, df_mean, df_std], axis=1)
    logger.info(f"  ok, {df.shape[1]} spatial features are added.")
    return df


def get_neighbors(array, row, col, radius=1):
    height, width = array.shape[0], array.shape[1]
    if row <= radius - 1:
        row_start, row_end = 0, row + radius + 1
    elif row >= height - radius:
        row_start, row_end = row - radius, None
    else:
        row_start, row_end = row - radius, row + radius + 1
    if col <= radius - 1:
        col_start, col_end = 0, col + radius + 1
    elif col == width - radius:
        col_start, col_end = col - radius, None
    else:
        col_start, col_end = col - radius, col + radius + 1
    # TODO: return neighbors + the point itself
    neighbors = array[row_start:row_end, col_start:col_end]
    return neighbors
