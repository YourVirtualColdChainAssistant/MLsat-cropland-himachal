import os
import re
import argparse


def main(args):
    download_raw_data(args.query, args.username, args.password,
                      args.query_filename, args.raw_data_dir)


def download_raw_data(query, user, pwd, query_fn, raw_data_dir):
    query_from_web(query, user, pwd, query_fn)
    ids = extract_products(query_fn)
    download(ids, user, pwd, raw_data_dir)


def query_from_web(query, user, pwd, query_fn):
    url_query = "https://scihub.copernicus.eu/dhus/odata/v1/Products" + query
    cmd_query = f'wget --no-check-certificate --user={user} --password={pwd} \
                    --output-document={query_fn} "{url_query}"'
    if not os.path.isfile(query_fn):
        os.system(cmd_query)


def extract_products(input_doc):
    with open(input_doc) as f:
        lines = f.readlines()
    query_data = ''
    for line in lines:
        query_data += line
    # extract Id, name, online status
    products = re.findall('<m:properties><d:Id>(.*)</d:Id><d:Name>(.*)</d:Name>.*<d:Online>(.*)</d:Online>', query_data)
    return products


def download(products, user, pwd, raw_data_dir):
    # only download online data
    for product in products:
        product_id, product_name, product_online = product
        url_download = f"https://scihub.copernicus.eu/dhus/odata/v1/Products('{product_id}')"
        # check name
        if product_name.startswith('S2A_MSIL1C'):
            cmd_download = f'wget --content-disposition --continue --user={user} --password={pwd} \
                        "{url_download}/$value" -P "{raw_data_dir}"'
            os.system(cmd_download)
    print("Successfully download all required data!")
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
    parser.add_argument('--raw_data_dir', type=str,
                        default='N:/dataorg-datasets/sentinel2_images/images_danya/raw/')
    args = parser.parse_args()
    main(args)
