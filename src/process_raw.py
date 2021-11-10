import os
import re
import glob
import shutil
import zipfile
import argparse
import shapely
import numpy as np
import geopandas as gpd
from sentinelsat import SentinelAPI

import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge

from util import clip_raster, convert_gml_to_shp


def main(args):
    # download_raw(args.user, args.password, args.images_dir)
    process_raw(args.images_dir)
    # clip_raster(args.images_dir, clip_from_shp='../data/study_area/study_area.shp')
    # clip_raster(args.images_dir, clip_from_shp='../data/train_region/train_region.shp')
    # clip_raster(args.images_dir, clip_from_shp='../data/test_region_near/test_region_near.shp')


def download_raw(user, pwd, images_dir):
    # raw directory
    raw_dir = images_dir + 'raw/'
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)

    # connect to API
    api = SentinelAPI(user, pwd)

    # search by polygon, time, and Hub query keywords
    # polygon = geojson.Polygon(polygon) # can download by polygon region
    products = api.query(date=('20200801', '20200831'),
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
    api.download_all(products, directory_path=raw_dir)

    print('Downloaded all the required data!')


def process_raw(images_dir):
    raw_dir = images_dir + 'raw/'
    safe_dir = images_dir + 'safe/'
    cloud_dir = images_dir + 'cloud_mask/'
    corrected_dir = images_dir + 'corrected/'
    geotiff_dir = images_dir + 'geotiff/'
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)
    if not os.path.exists(safe_dir):
        os.mkdir(safe_dir)
    if not os.path.exists(cloud_dir):
        os.mkdir(cloud_dir)
    if not os.path.exists(corrected_dir):
        os.mkdir(corrected_dir)
    if not os.path.exists(geotiff_dir):
        os.mkdir(geotiff_dir)

    # unzip files
    # unzip_products(raw_dir, safe_dir)

    # cloud mask
    # convert_gml_to_shp(safe_dir, cloud_dir)

    # sen2cor
    # sen2cor_path = 'C:\\Users\\lida\\Downloads\\Sen2Cor-02.09.00-win64\\L2A_Process.bat'
    # atmospheric_correction(sen2cor_path, safe_dir, corrected_dir)  # absolute path to call sen2cor

    # merge to single raster
    merge_to_raster(corrected_dir, geotiff_dir, cloud_dir)


def unzip_products(raw_dir, safe_dir):
    raw_files = os.listdir(raw_dir)
    for i, raw_file in enumerate(raw_files, start=1):
        raw_file_dir = raw_dir + raw_file
        with zipfile.ZipFile(raw_file_dir, 'r') as zip_file:
            print(f"[{i}/{len(raw_files)}] Unzipping {raw_file_dir}")
            zip_file.extractall(safe_dir)
    print("Unzip done!")


def atmospheric_correction(sen2cor_path, safe_dir, corrected_dir):
    safe_files = os.listdir(safe_dir)
    # safe_files = ['S2A_MSIL1C_20210502T053641_N0300_R005_T43SFR_20210502T074642.SAFE']
    for i, safe_file in enumerate(safe_files, start=1):
        safe_file_dir = safe_dir + safe_file
        if is_corrected(safe_file_dir, corrected_dir):
            print(f"[{i}/{len(safe_files)}] {safe_file_dir} corrected!")
        else:
            print(f"[{i}/{len(safe_files)}] Correcting {safe_file_dir}")
            os.system(f"{sen2cor_path} {safe_file_dir} --output_dir {corrected_dir}")
    print("Correction done!")


def is_corrected(safe_dir_to_correct, corrected_dir):
    # filename inside the GRANULE folder in safe folder
    name_to_correct = os.listdir(safe_dir_to_correct + '/GRANULE/')[0]
    # check what's in the corrected folder one by one
    for safe_corrected in os.listdir(corrected_dir):
        # filename inside the GRANULE folder in corrected folder
        name_corrected = os.listdir(corrected_dir + safe_corrected + '/GRANULE/')[0]
        path_corrected = corrected_dir + safe_corrected + '/GRANULE/' + name_corrected + '/IMG_DATA/'
        # check if two folder names are the same
        flag_name = name_to_correct.split('_')[1:] == name_corrected.split('_')[1:]
        if not flag_name:
            continue
        # TODO: fail if no file in GRANULE...
        # check file existence
        res_expected_num = [7, 13, 15]
        flag_res = True
        for i, res in enumerate(['R10m/', 'R20m/', 'R60m/']):
            if os.path.exists(path_corrected + res):
                if len(os.listdir(path_corrected + res)) != res_expected_num[i]:
                    # file number doesn't match
                    flag_res = False
                    shutil.rmtree(corrected_dir + safe_corrected)
                    print(f'Deleted {corrected_dir + safe_corrected}')
                    break
            else:
                # resolution files are not existent
                flag_res = False
                shutil.rmtree(corrected_dir + safe_corrected)
                print(f'Deleted {corrected_dir + safe_corrected}')
                break
        if flag_res:
            return True
    return False


def to_single_raster(input_dir, output_dir, cloud_mask_dir):
    """ Convert a folder of multiple geotiff images as a single multi-band geotiff
            INPUT : input_dir (str) -> path to the input folder as (path_to_folder/../*.tiff)
                    output_dir (str) -> path to where the geotiff will be saved (path_to_folder/../image_name.tiff)
            OUTPUT : None
        """
    # order is fixed by `sorted`
    file_path = [f for f in sorted(glob.glob(input_dir))]
    cloud_legend = {'CLOUDLESS': 0, 'OPAQUE': 1, 'CIRRUS': 2}
    # Read metadata of first file
    with rasterio.open(file_path[0]) as src0:
        meta = src0.meta
    # Update meta to reflect the number of layers
    meta.update(count=len(file_path) + 1)
    # Read each layer and write it to stack
    with rasterio.open(output_dir, 'w', **meta) as dst:
        for i, path in enumerate(file_path, start=1):
            with rasterio.open(path) as src1:
                dst.write_band(i, src1.read(1))
        if os.path.exists(cloud_mask_dir):
            cloud_shp = gpd.read_file(cloud_mask_dir)
            shapes = iter([(shapely.geometry.mapping(poly), cloud_legend[v]) for poly, v in
                           zip(cloud_shp.geometry, cloud_shp.maskType)])
            cloud_img = rasterio.features.rasterize(shapes, out_shape=(meta['height'], meta['width']),
                                                    transform=meta['transform'])
        else:  # cloud_mask_dir is empty (converting from gml failed or the whole tile is cloudless)
            cloud_img = np.zeros((meta['height'], meta['width']))
        dst.write_band(i + 1, cloud_img)


def merge_to_raster(corrected_dir, geotiff_dir, cloud_dir):
    file_paths = [f for f in os.listdir(corrected_dir)]
    print('Merging raster...')
    # save to single geotiff
    for i, file_path in enumerate(file_paths, start=1):
        input_dir = corrected_dir + file_path + '/GRANULE/'
        file_name = os.listdir(input_dir)[0]
        input_dir += file_name + '/IMG_DATA/R10m/*_B*.jp2'
        output_dir = geotiff_dir + file_name + '.tiff'
        cloud_mask_dir = cloud_dir + 'L1C' + file_name.lstrip('L2A') + '/cloud_mask.shp'
        to_single_raster(input_dir, output_dir, cloud_mask_dir)
        print(f'[{i}/{len(file_paths)}] merged {output_dir}')
    print('Merge done!')


def mask_raster(tiff_name, proj=None):
    """
    Clip and mask the raster with given projection.

    :param tiff_name: [str] name to open
    :param proj: [shp] given projection
    :return:
    """
    if proj is not None:
        with rasterio.open(tiff_name) as src:
            out_image, out_transform = rasterio.mask.mask(src, proj.geometry, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})

        masked_name = masked_name = tiff_name.split('.')[0] + '_masked.' + tiff_name.split('.')[1]
        with rasterio.open(masked_name, "w", **out_meta) as dest:
            dest.write(out_image)
            print(f'Write {masked_name}.')
    else:
        print('No projection for masking.')


# not using
def merge_tiles(img_ds, f1, f2, t1, t2):
    def create_dataset(data, crs, transform):
        # Receives a 2D array, a transform and a crs to create a rasterio dataset
        mem_file = MemoryFile()
        dataset = mem_file.open(driver='GTiff', height=data.shape[0], width=data.shape[1],
                                count=1, crs=crs, transform=transform, dtype=data.dtype)
        dataset.write(data, 1)

        return dataset

    file1 = create_dataset(f1[0], img_ds['B2'].profile['crs'], t1)
    file2 = create_dataset(f2[0], img_ds['B2'].profile['crs'], t2)

    merged, transform = merge([file1, file2])

    return merged, transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # authentication
    parser.add_argument(
        '--user',
        type=str,
        default='danyayay')
    parser.add_argument(
        '--password',
        type=str,
        default='empa.401')
    parser.add_argument(
        '--images_dir',
        type=str,
        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/')
    args = parser.parse_args()
    main(args)
