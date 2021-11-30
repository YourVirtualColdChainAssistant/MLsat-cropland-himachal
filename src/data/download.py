import os
import datetime
import argparse
from sentinelsat import SentinelAPI


def download(args):
    for m in range(4, 5):
        date_range = get_month_first_last_date(m)
        download_date(args.user, args.password, args.img_dir, args.tile_id, date_range)
    print('Downloaded all the required data!')


def download_date(user, pwd, img_dir, tile_id, date_range):
    # update img_dir
    img_dir = img_dir + tile_id + '/'

    # raw directory
    raw_dir = img_dir + 'raw/'
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)

    # connect to API
    api = SentinelAPI(user, pwd)

    # search by polygon, time, and Hub query keywords
    # polygon = geojson.Polygon(polygon) # can download by polygon region
    products_L1C = api.query(date=(date_range[0], date_range[1]),
                             platformname='Sentinel-2',
                             processinglevel='Level-1C',
                             raw=f'tileid:{tile_id}')
    products_L2A = api.query(date=(date_range[0], date_range[1]),
                             platformname='Sentinel-2',
                             processinglevel='Level-2A',
                             raw=f'tileid:{tile_id}')
    if len(products_L2A) == len(products_L1C):
        products = products_L2A
    else:
        products = products_L1C
    print(f'Found {len(products)} products in {date_range}')

    # check the number of online and offline products
    off_nb = 0
    for p_id in products.keys():
        p_info = api.get_product_odata(p_id)
        if not p_info['Online']:
            off_nb += 1
    print(f'{len(products) - off_nb} online + {off_nb} offline products')

    # download all results from the search
    api.download_all(products, directory_path=raw_dir)

    print(f'Downloaded data from {date_range[0]} to {date_range[1]}!')


def get_month_first_last_date(month, year=2020):
    first = datetime.date(year, month, 1)
    if month != 12:
        last = first.replace(month=month + 1) - datetime.timedelta(days=1)
    else:  # month = 12
        last = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
    return first.strftime("%Y%m%d"), last.strftime("%Y%m%d")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # authentication
    parser.add_argument('--user', type=str, default='danyayay')
    parser.add_argument('--password', type=str, default='empa.401')
    parser.add_argument('--img_dir', type=str,
                        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/')
    parser.add_argument('--tile_id', type=str, default='43RGQ')
    args = parser.parse_args()

    download(args)
