import os
import glob
import shutil
import zipfile
import argparse
import rasterio
import numpy as np
from src.utils.util import resample, find_file, find_folder, find_top_level
import copy
import multiprocessing


def process(args):
    print(args)
    if args.work_station:
        img_dir = '/mnt/N/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/'
    else:
        img_dir = 'N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/'
    tile_dir = img_dir + args.tile_id + '/'
    raw_dir = tile_dir + 'raw/'
    corrected_dir = tile_dir + 'L2A/'
    geotiff_dir = tile_dir + 'raster/'
    if not os.path.exists(corrected_dir):
        os.mkdir(corrected_dir)
    if not os.path.exists(geotiff_dir):
        os.mkdir(geotiff_dir)

    if args.work_station:
        sen2cor_path = 'sudo /home/lida/Documents/Sen2Cor-02.09.00-Linux64/bin/L2A_Process'
    else:
        sen2cor_path = 'C:\\Users\\lida\\Downloads\\Sen2Cor-02.09.00-win64\\L2A_Process.bat'
        # sen2cor_path = 'N:\\dataorg-datasets\\MLsatellite\\Sen2Cor-02.09.00-win64\\L2A_Process.bat'
    # check which level we downloaded
    processing_level = os.listdir(raw_dir)[0].split('_')[1][3:]
    if processing_level == 'L1C':
        safe_dir = tile_dir + 'L1C/'
        if not os.path.exists(safe_dir):
            os.mkdir(safe_dir)
        unzip_products(raw_dir, safe_dir, args.store_inter)
        atmospheric_correction(sen2cor_path, safe_dir, corrected_dir, args.store_inter)  # absolute path to call sen2cor
    else:
        unzip_products(raw_dir, corrected_dir, args.store_inter)
    merge_to_raster(corrected_dir, geotiff_dir)


def unzip_products(raw_dir, safe_dir, store_inter):
    raw_files = os.listdir(raw_dir)
    for i, raw_file in enumerate(raw_files, start=1):
        # unzip
        raw_file_dir = raw_dir + raw_file
        with zipfile.ZipFile(raw_file_dir, 'r') as zip_file:
            print(f"[{i}/{len(raw_files)}] Unzipping {raw_file_dir}")
            zip_file.extractall(safe_dir)
            if not store_inter:
                shutil.rmtree(raw_file_dir)
    print("Unzip done!")


def atmospheric_correction(sen2cor_path, safe_dir, corrected_dir, store_inter):
    safe_files = os.listdir(safe_dir)
    for i, safe_file in enumerate(safe_files, start=1):
        safe_file_dir = safe_dir + safe_file
        if is_corrected(safe_file_dir, corrected_dir):
            print(f"[{i}/{len(safe_files)}] {safe_file_dir} corrected!")
        else:
            print(f"[{i}/{len(safe_files)}] Correcting {safe_file_dir}")
            os.system(f'{sen2cor_path} {safe_file_dir} --output_dir {corrected_dir}')
            if not store_inter:
                shutil.rmtree(safe_file_dir)
    print("Correction done!")


def is_corrected(file_dir_to_correct, corrected_dir):
    name_L2A = os.listdir(file_dir_to_correct + '/GRANULE/')[0].replace('L1C', 'L2A')
    folders = find_top_level(name_L2A.split('_')[-1].split('T')[0], corrected_dir)
    result = []
    for folder in folders:
        result = result + find_folder(name_L2A, folder)
    if len(result) == 0:
        return False
    elif len(result) == 1:
        folder_in = result[0]
        corrected = False
        if set(os.listdir(folder_in)) == {'IMG_DATA', 'MTD_TL.xml', 'QI_DATA', 'AUX_DATA'}:
            path_L2A = folder_in + '/IMG_DATA/'
            if len(os.listdir(path_L2A)) == 3:
                n_expected = [7, 13, 15]
                all_exist = True
                for i, resolution in enumerate(['R10m/', 'R20m/', 'R60m/']):
                    if len(os.listdir(path_L2A + resolution)) != n_expected[i]:
                        all_exist = False
                        break
                if all_exist:
                    corrected = True
        if not corrected:
            folder_top = folder_in.split('GRANULE')[0]
            shutil.rmtree(folder_top)
            print(f'Deleted {folder_top}')
        return corrected
    else:
        for folder_in in result:
            folder_top = folder_in.split('GRANULE')[0]
            shutil.rmtree(folder_top)
        print(f'Deleted {result}')
        return False


def merge_to_raster(corrected_dir, geotiff_dir):
    # get crs first
    crs = get_crs_from_SCL(corrected_dir)
    print(f'crs = {crs}')

    # list files
    file_paths = [f for f in os.listdir(corrected_dir)]
    print('Merging raster...')
    # save to single geotiff
    for i, file_path in enumerate(file_paths, start=1):
        in_dir = corrected_dir + file_path + '/GRANULE/'
        file_name = os.listdir(in_dir)[0]
        in_path = in_dir + file_name + '/IMG_DATA/R10m/*_B*.jp2'
        out_path = geotiff_dir + file_name + '.tiff'
        cloud_path = find_file('SCL_60m.jp2', in_dir)
        convert_jp2_to_tiff(in_path, out_path, cloud_path, crs)
        print(f'[{i}/{len(file_paths)}] merged {out_path}')
    print('Merge done!')


def convert_jp2_to_tiff(in_path, out_path, cloud_path, crs):
    """
    Convert a folder of multiple jp2 images as a single multi-band geotiff

    Parameters
    ----------
    in_path: string
        path to the input folder as (path_to_folder/../*.jp2)
    out_path: string
        path to where the geotiff will be saved (path_to_folder/../out_name.tiff)
    cloud_path: A list of string
        paths founds with SCL
    crs: crs
        coordinate system of the target tiff

    Returns
    -------

    """
    # order is fixed by `sorted`
    jp2_path = [f for f in sorted(glob.glob(in_path))]
    # legend
    bands_dcp = [file.split('_')[-2] for file in jp2_path] + ['cloud mask']
    # Read metadata of first file
    with rasterio.open(jp2_path[0]) as src0:
        meta = src0.meta

    # prepare cloud mask
    if len(cloud_path) == 1:
        cloud_img, _ = resample(cloud_path[0], meta['height'], meta['width'])
        cloud_img = np.squeeze(cloud_img)
    elif len(cloud_path) == 0:
        raise ValueError('No SCL file found.')
    else:
        raise ValueError('Found multiple SCL files.')

    # Update meta to reflect the number of layers
    meta.update(count=len(jp2_path) + 1, driver='GTiff', crs=crs)
    # crs=rasterio.crs.CRS.from_user_input(pyproj.CRS('EPSG:32643')))
    # Read each layer and write it to stack
    with rasterio.open(out_path, 'w', **meta) as dst:
        # stack four bands
        for i, path in enumerate(jp2_path, start=1):
            with rasterio.open(path) as src:
                dst.write_band(i, src.read(1))
                dst.set_band_description(i, bands_dcp[i - 1])
        # stack cloud mask
        dst.write_band(i + 1, cloud_img)
        dst.set_band_description(i + 1, bands_dcp[i])


def get_crs_from_SCL(corrected_dir):
    corrected_names = os.listdir(corrected_dir)
    for corrected_name in corrected_names:
        file = find_file('SCL_60m.jp2', corrected_dir + corrected_name)
        if len(file) == 1:
            data = rasterio.open(file[0])
            return data.crs
        else:
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_station', type=bool, default=False)
    parser.add_argument('--store_inter', default=True, action='store_false',
                        help='Store the intermediate files (raw and L1C) or not.')
    parser.add_argument('--tile_ids', nargs="+", default=['43RGP'])
    args = parser.parse_args()

    args_list, tile_ids = [], args.tile_ids
    print(f'Parallizing to {len(tile_ids)} processes...')
    for tile_id in tile_ids:
        args.tile_id = tile_id
        args_list.append(copy.deepcopy(args))  # deep copy 
    process_pool = multiprocessing.Pool(processes=len(tile_ids))
    process_pool.map(process, args_list)
