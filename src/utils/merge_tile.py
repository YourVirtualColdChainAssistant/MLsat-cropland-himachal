import glob
import argparse
from rasterio.merge import merge


def merge_tile_predictions(args):
    if args.workstation:
        img_dir = '/mnt/N/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/'
    else:
        img_dir = 'N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/'
    pred_path = img_dir + 'predictions/' + args.pretrained + '/'
    pred_list = [f for f in glob.glob(pred_path + '*.tiff')]
    merge(pred_list, dst_path=pred_path + '../' + args.pretrained + '.tiff', method='max')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workstation', type=bool, default=True)
    parser.add_argument('--pretrained', type=str, default='20220105-135132_rfc')
    parser.add_argument('--method', type=str, default='max')

    args = parser.parse_args()

    merge_tile_predictions(args)
