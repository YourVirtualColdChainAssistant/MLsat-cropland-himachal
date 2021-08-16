import os
import re
import glob
import shutil
import zipfile
import argparse
import rasterio


def main(args):
    process_raw(args.images_dir)


def process_raw(images_dir):
    # unzip files
    raw_dir = images_dir + 'raw/'
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)
    # unzip_products(raw_dir)

    # L1C to L2A
    corrected_dir = images_dir + 'corrected/'
    if not os.path.exists(corrected_dir):
        os.mkdir(corrected_dir)
    # absolute path to call sen2cor
    sen2cor_loc = 'C:\\Users\\lida\\Downloads\\Sen2Cor-02.09.00-win64\\L2A_Process.bat'
    atmospheric_correction(sen2cor_loc, raw_dir, corrected_dir, args.resolution)

    # merge to single raster
    geotiff_dir = images_dir + 'geotiff/'
    if not os.path.exists(geotiff_dir):
        os.mkdir(geotiff_dir)
    # raster_converter(corrected_dir, geotiff_dir)


def unzip_products(raw_data_dir):
    raw_files = [f for f in os.listdir(raw_data_dir) if re.match('.*zip', f)]
    for i, raw_file in enumerate(raw_files, start=1):
        raw_file_dir = raw_data_dir + raw_file
        with zipfile.ZipFile(raw_file_dir, 'r') as zip_file:
            print(f"[{i}/{len(raw_files)}] Unzipping {raw_file_dir}")
            zip_file.extractall(raw_data_dir)
    print("Unzip done!")


def atmospheric_correction(sen2cor_loc, raw_dir, corrected_dir, resolution=None):
    safe_files = [f for f in os.listdir(raw_dir) if re.match('.*SAFE', f)]
    # safe_files = ['S2A_MSIL1C_20210502T053641_N0300_R005_T43SFR_20210502T074642.SAFE']
    for i, safe_file in enumerate(safe_files, start=1):
        safe_file_dir = raw_dir + safe_file
        if is_corrected(safe_file_dir, corrected_dir):
            print(f"[{i}/{len(safe_files)}] {safe_file_dir} corrected!")
        else:
            print(f"[{i}/{len(safe_files)}] Correcting {safe_file_dir}")
            # output tif rather than jp2
            if resolution is None:
                os.system(f"{sen2cor_loc} {safe_file_dir} --output_dir {corrected_dir} --tif")
            else:
                os.system(f"{sen2cor_loc} {safe_file_dir} --output_dir {corrected_dir} --resolution {resolution} --tif")
    print("Correction done!")


def is_corrected(safe_dir_to_correct, corrected_dir):
    name_to_correct = os.listdir(safe_dir_to_correct + '/GRANULE/')[0]
    # check what's in the corrected folder
    for safe_corrected in os.listdir(corrected_dir):
        name_corrected = os.listdir(corrected_dir + safe_corrected + '/GRANULE/')[0]
        path_corrected = corrected_dir + safe_corrected + '/GRANULE/' + name_corrected
        flag_name = name_to_correct.split('_')[1:] == name_corrected.split('_')[1:]
        # TODO: fail if no file in GRANULE...
        tmp_files = os.listdir(path_corrected)
        flag_tmp = False
        for f in tmp_files:
            if f.startswith('tmp'):
                flag_tmp = True
                shutil.rmtree(corrected_dir + safe_corrected)
                print(f'Deleted {corrected_dir + safe_corrected}')
                break
        # if flag name already exists and no temporary files, then the correction is done.
        if flag_name and not flag_tmp:
            return True
    return False


def merge_to_single_raster(input_dir, output_dir):
    """ Convert a folder of multiple geotiff images as a single multi-band geotiff
            INPUT : input_dir (str) -> path to the input folder as (path_to_folder/../*.tiff)
                    output_dir (str) -> path to where the geotiff will be saved (path_to_folder/../image_name.tiff)
            OUTPUT : None
        """
    file_path = [f for f in sorted(glob.glob(input_dir))]
    # Read metadata of first file
    with rasterio.open(file_path[0]) as src0:
        meta = src0.meta
    # Update meta to reflect the number of layers
    meta.update(count=len(file_path))
    # Read each layer and write it to stack
    with rasterio.open(output_dir, 'w', **meta) as dst:
        for i, path in enumerate(file_path, start=1):
            with rasterio.open(path) as src1:
                dst.write_band(i, src1.read(1))


def raster_converter(corrected_dir, geotiff_dir):
    file_paths = [f for f in os.listdir(corrected_dir)]
    # save to single geotiff
    for file_path in file_paths:
        input_dir = corrected_dir + file_path + '/GRANULE/'
        file_name = os.listdir(input_dir)[0]
        input_dir += file_name + '/IMG_DATA/R10m/*.tif'
        output_dir = geotiff_dir + file_name + '.tiff'
        # merge_to_single_raster(input_dir, output_dir)
        print(f'Saved {output_dir}')


def merge_tiles():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # configuration
    parser.add_argument(
        '--images_dir',
        type=str,
        default='N:/dataorg-datasets/sentinel2_images/images_danya/'
    )
    parser.add_argument(
        '--resolution',
        default=None
    )
    args = parser.parse_args()
    main(args)
