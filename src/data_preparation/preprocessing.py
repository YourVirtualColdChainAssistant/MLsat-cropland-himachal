import argparse

def calculate_ndvi(red, nir):
    """ Compute the NDVI
        INPUT : red (np.array) -> the Red band images as a numpy array of float
                nir (np.array) -> the Near Infrared images as a numpy array of float
        OUTPUT : ndvi (np.array) -> the NDVI
    """
    ndvi = (nir - red) / (nir + red + 1e-12)
    return ndvi


def select_bands():
    pass


def feature_engineering():
    pass


def preprocessing():
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