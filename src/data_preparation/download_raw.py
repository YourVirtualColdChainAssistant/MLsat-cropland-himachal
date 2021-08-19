import os
import re
import argparse

import geojson
from sentinelsat import SentinelAPI



def main(args):
    download_raw_data(args.user, args.password, args.raw_data_dir)


def download_raw_data(user, pwd, raw_data_dir):
    # connect to API
    api = SentinelAPI(user, pwd)

    # search by polygon, time, and Hub query keywords
    # polygon = geojson.Polygon(polygon) # can download by polygon region
    products = api.query(date=('20200101', '20200131'),
                         platformname='Sentinel-2',
                         processinglevel='Level-1C',
                         raw='tileid:43SFR',
                         )
    print(f'Find {len(products)} products!')

    # check the number of online and offline products
    off_nb = 0
    for p_id in products.keys():
        p_info = api.get_product_odata(p_id)
        if not p_info['Online']:
            off_nb += 1
    print(f'{len(products) - off_nb} online + {off_nb} offline products')

    # download all results from the search
    api.download_all(products, directory_path=raw_data_dir)

    print('Downloaded all the required data!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # authentication
    parser.add_argument('--user', type=str,
                        default='danyayay')
    parser.add_argument('--password', type=str,
                        default='empa.401')
    parser.add_argument('--raw_data_dir', type=str,
                        default='N:/dataorg-datasets/sentinel2_images/images_danya/raw/')
    args = parser.parse_args()
    main(args)
