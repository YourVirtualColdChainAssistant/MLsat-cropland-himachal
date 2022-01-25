import shapely
import argparse
import rasterio
import geopandas as gpd
from src.data.clip import clip_single_raster


def output_districts(work_station, filename):
    # district file
    india_adm = './data/india_adm/IND_adm2 - HP.shp'
    district_shp = gpd.read_file(india_adm)

    # predictions
    if work_station:
        img_dir = '/mnt/N/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/'
    else:
        img_dir = 'N:/dataorg-datasets/MLsatellite/sentinel2_images/images_danya/'
    in_path = img_dir + 'predictions/cropland_preds/' + filename + '.tiff'
    with rasterio.open(in_path, 'r') as src:
        in_crs = src.crs
    if district_shp.crs != in_crs:
        district_shp = district_shp.to_crs(in_crs)

    # mask to districts
    d_shp = district_shp[(district_shp['NAME_2'] == 'Kullu') | (district_shp['NAME_2'] == 'Mandi') | (district_shp['NAME_2'] == 'Shimla')]
    d_shapes = [shapely.geometry.mapping(s) for s in d_shp.geometry if s is not None]
    d_path = img_dir + 'predictions/cropland_preds/cropland_pred_colored.tiff'
    clip_single_raster(d_shapes, in_path, d_path, updates={'nodata': -9999})
    clip_single_raster(d_shapes, in_path, d_path, updates={'compress': 'lzw', 'nodata': -9999})
    print(f'saved to {d_path}')
    g_path = img_dir + f'predictions/cropland_preds/cropland_pred_general.tiff'
    generate_general_map(d_path, g_path, updates={'nodata': -1})
    generate_general_map(d_path, g_path, updates={'compress': 'lzw', 'nodata': -1})
    print(f'saved to {d_path}')


def generate_general_map(in_path, out_path, updates=None):
    with rasterio.open(in_path, 'r') as src:
        pred = src.read(1)
        meta = src.meta
    pred[(pred != -9999) & (pred != -9998)] = 1
    pred[pred == -9998] = 0
    pred[pred == -9999] = -1

    meta.update({'dtype': rasterio.int8})
    if updates:
        meta.update(updates)
        if 'compress' in updates.keys():
            out_path = out_path.replace('.tiff', '_lzw.tiff')

    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(pred.astype(rasterio.int8), 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workstation', type=bool, default=True)
    parser.add_argument('--filename', type=str, default='20220105-135132_rfc')
    args = parser.parse_args()

    output_districts(args.workstation, args.filename)
