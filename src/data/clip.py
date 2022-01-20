import os
import shapely
import rasterio
import geopandas as gpd
from rasterio.mask import mask


def clip_raster(img_dir, clip_from_shp):
    # check directory
    geotiff_dir = img_dir + 'geotiff/'
    clip_dir = img_dir + clip_from_shp.split('/')[2] + '/'
    if not os.path.exists(clip_dir):
        os.mkdir(clip_dir)

    # get all files
    clip_names = [f for f in sorted(os.listdir(geotiff_dir)) if f.endswith('.tiff')]

    # get the target crs (raster's crs rather than shp's crs)
    with rasterio.open(geotiff_dir + clip_names[0]) as src0:
        raster_crs = src0.crs
    # get the shp crs
    shp = gpd.read_file(clip_from_shp)
    # reproject shp if needed
    if shp.crs != raster_crs:
        shp = reproject_shapefile(raster_crs, shp)
    # get geojson-like shapes
    shapes = [shapely.geometry.mapping(s) for s in shp.geometry if s is not None]

    print('\nStart clipping...')
    for i, clip_name in enumerate(clip_names, start=1):
        geotiff_path = geotiff_dir + clip_name
        clip_path = clip_dir + clip_name
        clip_flag = clip_single_raster(shapes, geotiff_path, clip_path)
        if clip_flag:
            print(f'[{i}/{len(clip_names)}] Clipped {clip_path}')
        else:
            print(f'[{i}/{len(clip_names)}] Discarded {clip_path}')
    print(f'Clip done!')


def clip_single_raster(shapes, geotiff_path, clip_path):
    """

    :param shapes: geometry of source shapefile
    :param geotiff_path: xx.tiff path
    :param clip_path: output clipped path
    :return:
    """
    # read imagery file
    with rasterio.open(geotiff_path) as src:
        out_image, out_transform = mask(src, shapes, crop=True, all_touched=True)
        out_meta = src.meta

    # Save clipped imagery
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    clip_flag = False
    if out_image.mean() != 0.0:
        with rasterio.open(clip_path, "w", **out_meta) as dst:
            dst.write(out_image)
        clip_flag = True
    return clip_flag


def reproject_shapefile(dst_crs, shp):
    return shp.to_crs(dst_crs)
