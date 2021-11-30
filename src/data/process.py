import os
import glob
import shutil
import zipfile
import argparse
import shapely
import numpy as np
import geopandas as gpd
import rasterio


def process(args):
    tile_dir = args.img_dir + args.tile_id + '/'
    raw_dir = tile_dir + 'raw/'
    cloud_dir = tile_dir + 'cloud_mask/'
    corrected_dir = tile_dir + 'L2A/'
    geotiff_dir = tile_dir + 'raster/'
    if not os.path.exists(cloud_dir):
        os.mkdir(cloud_dir)
    if not os.path.exists(corrected_dir):
        os.mkdir(corrected_dir)
    if not os.path.exists(geotiff_dir):
        os.mkdir(geotiff_dir)
    sen2cor_path = 'C:\\Users\\lida\\Downloads\\Sen2Cor-02.09.00-win64\\L2A_Process.bat'
    # sen2cor_path = '../../Sen2Cor-02.09.00-win64/L2A_Process.bat'
    # check which level we downloaded
    processing_level = os.listdir(raw_dir)[0].split('_')[1][3:]
    if processing_level == 'L1C':
        safe_dir = tile_dir + 'L1C/'
        if not os.path.exists(safe_dir):
            os.mkdir(safe_dir)
        unzip_products(raw_dir, safe_dir)
        get_cloud_mask(safe_dir, cloud_dir)
        atmospheric_correction(sen2cor_path, safe_dir, corrected_dir)  # absolute path to call sen2cor
    else:
        unzip_products(raw_dir, corrected_dir)
    merge_to_raster(corrected_dir, geotiff_dir, cloud_dir)


def unzip_products(raw_dir, safe_dir):
    raw_files = os.listdir(raw_dir)
    for i, raw_file in enumerate(raw_files, start=1):
        # unzip
        raw_file_dir = raw_dir + raw_file
        with zipfile.ZipFile(raw_file_dir, 'r') as zip_file:
            print(f"[{i}/{len(raw_files)}] Unzipping {raw_file_dir}")
            zip_file.extractall(safe_dir)
    print("Unzip done!")


def get_cloud_mask(safe_dir, cloud_dir):
    safe_names = os.listdir(safe_dir)
    print('Converting cloud mask from gml to shp ...')
    for i, safe_name in enumerate(safe_names, start=1):
        filename = os.listdir(safe_dir + safe_name + '/GRANULE/')[0]
        gml_path = safe_dir + safe_name + '/GRANULE/' + filename + '/QI_DATA/MSK_CLOUDS_B00.gml'
        cloud_path, val = convert_gml_to_shp(cloud_dir + filename, gml_path)
        if val == 0:
            print(f'[{i}/{len(safe_names)}] {filename} ')
        else:
            print(f'[{i}/{len(safe_names)}] {filename} failed / empty')
    print(f'Cloud mask done!')


def atmospheric_correction(sen2cor_path, safe_dir, corrected_dir):
    safe_files = os.listdir(safe_dir)
    for i, safe_file in enumerate(safe_files, start=1):
        safe_file_dir = safe_dir + safe_file
        if is_corrected(safe_file_dir, corrected_dir):
            print(f"[{i}/{len(safe_files)}] {safe_file_dir} corrected!")
        else:
            print(f"[{i}/{len(safe_files)}] Correcting {safe_file_dir}")
            os.system(f"{sen2cor_path} {safe_file_dir} --output_dir {corrected_dir}")
    print("Correction done!")


def convert_gml_to_shp(cloud_folder, gml_path):
    if not os.path.exists(cloud_folder):
        os.mkdir(cloud_folder)
    cloud_path = cloud_folder + '/cloud_mask.shp'
    val = os.system(f'ogr2ogr -f "ESRI Shapefile" {cloud_path} {gml_path}')
    return cloud_path, val


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


def merge_to_raster(corrected_dir, geotiff_dir, cloud_dir):
    # get crs first
    crs = get_crs_from_cloud_mask(cloud_dir)
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
        cloud_path = cloud_dir + 'L1C' + file_name.lstrip('L2A') + '/cloud_mask.shp'
        convert_jp2_to_tiff(in_path, cloud_path, out_path, crs)
        print(f'[{i}/{len(file_paths)}] merged {out_path}')
    print('Merge done!')


def convert_jp2_to_tiff(in_path, cloud_path, out_path, crs):
    """
    Convert a folder of multiple jp2 images as a single multi-band geotiff

    Parameters
    ----------
    in_path: string
        path to the input folder as (path_to_folder/../*.jp2)
    cloud_path: string
        path to cloud mask
    out_path: string
        path to where the geotiff will be saved (path_to_folder/../out_name.tiff)
    crs: crs
        coordinate system of the target tiff

    Returns
    -------

    """
    # order is fixed by `sorted`
    jp2_path = [f for f in sorted(glob.glob(in_path))]
    bands_dcp = [file.split('_')[-2] for file in jp2_path] + ['cloud mask']
    cloud_legend = {'CLOUDLESS': 0, 'OPAQUE': 1, 'CIRRUS': 2}

    # Read metadata of first file
    with rasterio.open(jp2_path[0]) as src0:
        meta = src0.meta

    # rasterize cloud mask
    if os.path.exists(cloud_path):
        cloud_shp = gpd.read_file(cloud_path)
        shapes = iter([(shapely.geometry.mapping(poly), cloud_legend[v]) for poly, v in
                       zip(cloud_shp.geometry, cloud_shp.maskType)])
        cloud_img = rasterio.features.rasterize(shapes, out_shape=(meta['height'], meta['width']),
                                                transform=meta['transform'])
    else:  # cloud_mask_dir is empty (converting from gml failed or the whole tile is cloudless)
        cloud_img = np.zeros((meta['height'], meta['width']))

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


def get_crs_from_cloud_mask(cloud_dir):
    cloud_names = os.listdir(cloud_dir)
    for cloud_name in cloud_names:
        cloud_path = cloud_dir + cloud_name + '/cloud_mask.shp'
        if os.path.exists(cloud_path):
            crs = gpd.read_file(cloud_path).crs
            return crs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', type=str,
                        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/')
    parser.add_argument('--tile_id', type=str, default='43RGQ')
    args = parser.parse_args()

    process(args)
