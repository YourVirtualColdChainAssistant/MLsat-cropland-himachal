import os
import glob
import shutil
import zipfile
import argparse
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge

def main(args):
    process_raw(args.images_dir)


def process_raw(images_dir):
    # unzip files
    raw_dir = images_dir + 'raw/'
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)
    # raw safe files
    safe_dir = images_dir + 'safe/'
    if not os.path.exists(safe_dir):
        os.mkdir(safe_dir)
    # unzip_products(raw_dir, safe_dir)

    # sen2cor
    corrected_dir = images_dir + 'corrected/'
    if not os.path.exists(corrected_dir):
        os.mkdir(corrected_dir)
    # absolute path to call sen2cor
    sen2cor_path = 'C:\\Users\\lida\\Downloads\\Sen2Cor-02.09.00-win64\\L2A_Process.bat'
    atmospheric_correction(sen2cor_path, safe_dir, corrected_dir)

    # merge to single raster
    geotiff_dir = images_dir + 'geotiff/'
    if not os.path.exists(geotiff_dir):
        os.mkdir(geotiff_dir)
    merge2single_raster(corrected_dir, geotiff_dir)


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


def to_single_raster(input_dir, output_dir):
    """ Convert a folder of multiple geotiff images as a single multi-band geotiff
            INPUT : input_dir (str) -> path to the input folder as (path_to_folder/../*.tiff)
                    output_dir (str) -> path to where the geotiff will be saved (path_to_folder/../image_name.tiff)
            OUTPUT : None
        """
    # order is fixed by `sorted`
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


def merge2single_raster(corrected_dir, geotiff_dir):
    file_paths = [f for f in os.listdir(corrected_dir)]
    # save to single geotiff
    for file_path in file_paths:
        input_dir = corrected_dir + file_path + '/GRANULE/'
        file_name = os.listdir(input_dir)[0]
        input_dir += file_name + '/IMG_DATA/R10m/*_B*.jp2'
        output_dir = geotiff_dir + file_name + '.tiff'
        to_single_raster(input_dir, output_dir)
        print(f'Saved {output_dir}')


def clip_raster(tiff_name, proj=None):
    """Clip the raster with given projection

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
        print('No projection for clipping.')


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
    # configuration
    parser.add_argument(
        '--images_dir',
        type=str,
        default='N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/'
    )
    args = parser.parse_args()
    main(args)
