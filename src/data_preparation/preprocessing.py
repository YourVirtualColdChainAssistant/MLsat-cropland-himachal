import argparse
import numpy as np


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


def preprocessing(img, add_features=['ndvi', 'ndre', 'gndvi']):
    new_bands = []

    # select specific bands
    img = img[:, :, [1, 2, 3, 4, 5]]

    # add feature
    for feature in add_features:
        if feature == 'ndvi':
            new_bands.append(calculate_ndvi(img[:, :, 1], img[:, :, 2]))
        elif feature == 'ndre':
            new_bands.append(calculate_ndre(img[:, :, 1], img[:, :, 2]))
        elif feature == 'gndvi':
            new_bands.append(calculate_gndvi(img[:, :, 1], img[:, :, 2]))

    return np.append(img, np.stack(new_bands, axis=2), axis=2)


def upsample():
    pass


def label():
    pass


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # configuration
    parser.add_argument('--images_dir', type=str,
                        default='N:/dataorg-datasets/sentinel2_images/images_danya/')
    args = parser.parse_args()
    main(args)