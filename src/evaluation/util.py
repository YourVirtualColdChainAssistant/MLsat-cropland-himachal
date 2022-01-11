import os
import pyproj
import shapely
import geopandas as gpd
import rasterio
from rasterio.windows import Window

from src.data.clip import clip_single_raster


def adjust_raster_size(dataset_path, out_path, region_indicator, meta, label_only=True):
    # clip to a bounding box, to reduce computation cost
    if isinstance(region_indicator, str):
        region_shp = gpd.read_file(region_indicator).to_crs(meta['crs'])
        minx, miny, maxx, maxy = region_shp.total_bounds
        box_shp = gpd.GeoDataFrame({'geometry': shapely.geometry.box(minx, miny, maxx, maxy)}, index=[0])
        box_shp = box_shp.set_crs(meta['crs'])
    elif isinstance(region_indicator, Window):
        minx, miny, maxx, maxy = rasterio.windows.bounds(region_indicator, meta['transform'])
        box_shp = gpd.GeoDataFrame({'geometry': shapely.geometry.box(minx, miny, maxx, maxy)}, index=[0])
        box_shp = box_shp.set_crs(meta['crs'])
    else:
        raise ValueError(f'Cannot adjust raster size by {region_indicator}')
    inter_path = out_path.replace(out_path.split('_')[-1], 'intermediate_result.tiff')
    clip_raster_to_shp(dataset_path, inter_path, box_shp)
    # align with same resolution
    align_raster(inter_path, out_path, meta, (minx, miny, maxx, maxy))
    if label_only:
        clip_raster_to_shp(out_path, out_path, region_shp)
    os.remove(inter_path)


def clip_raster_to_shp(in_path, out_path, region_shp):
    with rasterio.open(in_path, 'r') as src0:
        in_crs = src0.crs
    if region_shp.crs != in_crs:
        region_shp = region_shp.to_crs(in_crs)
    region_shapes = [shapely.geometry.mapping(s) for s in region_shp.geometry if s is not None]
    clip_single_raster(region_shapes, in_path, out_path)


def align_raster(in_path, out_path, meta, bounds):
    """
    Align according to prediction file (with boundary and resolution adjustment).
    -te {bounds.left} {bounds.bottom} {bounds.right} {bounds.top}
    bounds = (minx, miny, maxx, maxy)
    """
    # gdalwarp cannot support ogc_wkt, so convert to esri_wkt
    crs = pyproj.CRS.from_string(meta['crs'].to_string()).to_wkt(version='WKT1_ESRI')
    # command
    cmd = f"gdalwarp -overwrite -r average -t_srs {crs} -ts {meta['width']} {meta['height']} " + \
          f"-te {bounds[0]} {bounds[1]} {bounds[2]} {bounds[3]} {in_path} {out_path}"
    returned_val = os.system(cmd)
    if returned_val == 0:
        print('Aligned raster!')
    else:
        raise ValueError('Alignment failed!')
