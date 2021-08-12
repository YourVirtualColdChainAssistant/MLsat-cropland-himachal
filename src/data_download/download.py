import os
import re
import argparse


def prepare_data(args):
    query_from_web(args.query, args.username, args.password, args.query_filename)
    ids = extract_data_ids(args.query_filename)
    download_data(ids, args.username, args.password, args.sat_image_path)


def query_from_web(query, username, password, query_filename):
    url_query = "https://scihub.copernicus.eu/dhus/odata/v1/Products" + query
    cmd_query = f'wget --no-check-certificate --user={username} --password={password} \
                    --output-document={query_filename} "{url_query}"'
    if not os.path.isfile(query_filename):
        os.system(cmd_query)


def extract_data_ids(input_doc):
    with open(input_doc) as f:
        lines = f.readlines()
    query_data = ''
    for line in lines:
        query_data += line
    products = re.findall('<m:properties><d:Id>(.*)</d:Id><d:Name>(.*)</d:Name>.*<d:Online>(.*)</d:Online>', query_data)
    return products


def download_data(products, username, password, sat_image_path):
    # only download online data
    for product in products:
        product_id, product_name, product_online = product
        url_download = f"https://scihub.copernicus.eu/dhus/odata/v1/Products('{product_id}')"
        # check name
        if product_name.startswith('S2A_MSIL1C'):
            cmd_download = f'wget --content-disposition --continue --user={username} --password={password} \
                        "{url_download}/$value" -P "{sat_image_path}"'
            os.system(cmd_download)
    # for offline produce, we need to check online status when extracting ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # authentication
    parser.add_argument('--username', type=str,
                        default='danyayay')
    parser.add_argument('--password', type=str,
                        default='Ldy19970722')
    parser.add_argument('--query', type=str,
                        default='?$filter=substringof(%2743SFR%27,Name) and Online eq true &$top=100')
                        # startswith(Name,%20%27S2A_MSIL1C%27)%20and%20substringof(%272021%27,%20Name) &$top=100')
    parser.add_argument('--query_filename', type=str,
                        default='../../data/output100.txt')
    parser.add_argument('--sat_image_path', type=str,
                        default='N:/dataorg-datasets/sentinel2_images/images_danya/raw/')
    args = parser.parse_args()
    prepare_data(args)
