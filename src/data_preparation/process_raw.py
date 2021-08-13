import os
import re
import zipfile
import argparse

def main(args):
    process_raw(args.images_dir)


def process_raw(images_dir):
    # unzip files
    raw_data_dir = images_dir + 'raw/'
    if not os.path.exists(raw_data_dir):
        os.mkdir(raw_data_dir)
    # unzip_products(raw_data_dir)

    # L1C to L2A
    processed_data_dir = images_dir + 'processed/'
    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)
    # absolute path to call sen2cor
    sen2cor_loc = 'C:\\Users\\lida\\Downloads\\Sen2Cor-02.09.00-win64\\L2A_Process.bat'
    atmospheric_correction(sen2cor_loc, raw_data_dir, processed_data_dir, args.resolution)


def unzip_products(raw_data_dir):
    raw_files = [f for f in os.listdir(raw_data_dir) if re.match('.*zip', f)]
    for i, filename in enumerate(raw_files, start=1):
        raw_file = raw_data_dir + filename
        with zipfile.ZipFile(raw_file, 'r') as zip_file:
            print(f"[{i}/{len(raw_files)}] Unzipping {raw_file}")
            zip_file.extractall(raw_data_dir)
    print("Unzip done!")


def atmospheric_correction(sen2cor_loc, input_dir, output_dir, resolution=None):
    safe_files = [f for f in os.listdir(input_dir) if re.match('.*SAFE', f)]
    # safe_files = ['S2A_MSIL1C_20210502T053641_N0300_R005_T43SFR_20210502T074642.SAFE']
    for i, filename in enumerate(safe_files, start=1):
        safe_file = input_dir + filename
        print(f"[{i}/{len(safe_files)}] Correcting {safe_file}")
        # output tif rather than jp2
        if resolution is None:
            os.system(f"{sen2cor_loc} {safe_file} --output_dir {output_dir} --tif")
        else:
            os.system(f"{sen2cor_loc} {safe_file} --output_dir {output_dir} --resolution {resolution} --tif")
    print("Correction done!")


def merge2raster():
    pass


def merge_tiles():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # configuration
    parser.add_argument('--images_dir', type=str,
                        default='N:/dataorg-datasets/sentinel2_images/images_danya/')
    parser.add_argument('--resolution', default=None)
    args = parser.parse_args()
    main(args)