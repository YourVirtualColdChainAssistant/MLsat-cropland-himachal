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
    band02 = blue --> idx = 0
    band03 = green --> idx = 1
    band04 = red --> idx = 2
    band08 = nir --> idx = 3
    """
    new_bands = []

    # select specific bands
    # img = img[:, :, [1, 2, 3, 4]]

    # bands
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    nir = img[:, :, 3]

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

    return np.append(img, np.stack(new_bands, axis=2), axis=2)


def get_statistics(op, data, axis):
    if op == 'avg':
        return np.mean(data, axis=axis)
    elif op == 'std':
        return np.std(data, axis=axis)
    elif op == 'max':
        return np.max(data, axis=axis)
    else:
        return 'No corresponding calculation.'


def preprocessing(stacked_band):
    df = pd.DataFrame()
    for i, band in enumerate(['blue', 'green', 'red', 'nir']):
        for op in ['avg', 'std', 'max']:
            col_name = band + '_' + op
            reshaped_band = stacked_band[..., i].reshape(stacked_band[..., i].shape[0], -1)
            df[col_name] = get_statistics(op, reshaped_band, 0)
    return df
