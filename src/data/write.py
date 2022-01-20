import os
import rasterio
from src.evaluation.util import adjust_raster_size


def save_predictions_geotiff(predictions, save_path, meta,
                             region_indicator=None, color_by_height=False, hm_name='height_map'):
    """
    Save predictions into .tiff file.

    Parameters
    ----------
    predictions: np.array
        shape (height * width, )
    save_path: str
        path to save .tiff
    meta: dict
        meta data to save .tiff
    region_indicator: str or rasterio.window.Window
        an indicator about the region to save.
    color_by_height: bool
        Whether add a band of mask colored by height.
    hm_name: str
        height map file name to save

    Returns
    -------
    None
    """
    if color_by_height:
        # relative path of altitude data
        height_map_path = '../../../Data/layers_india/ancilliary_data/elevation/IND_alt.vrt'
        adjust_raster_size(height_map_path, f'./data/open_datasets/{hm_name}.tiff',
                           region_indicator=region_indicator, meta=meta, label_only=False)
        with rasterio.open(f'./data/open_datasets/{hm_name}.tiff') as f:
            height_map = f.read(1).reshape(-1)
        height_map[predictions != 2] = -9998
        with rasterio.Env():
            # Write an array as a raster band to a new 8-bit file. We start with the profile of the source
            meta.update(dtype=rasterio.int16, count=1)
            with rasterio.open(save_path, 'w', **meta) as dst:
                # reshape into (band, height, width)
                dst.write(height_map.reshape(meta['height'], meta['width']).astype(rasterio.int16), indexes=1)
        os.remove(f'./data/open_datasets/{hm_name}.tiff')
    else:
        color_map = {0: (0, 0, 0), 2: (154, 205, 50)}
        with rasterio.Env():
            # Write an array as a raster band to a new 8-bit file. We start with the profile of the source
            meta.update(dtype=rasterio.uint8, count=1)
            with rasterio.open(save_path, 'w', **meta) as dst:
                # reshape into (band, height, width)
                dst.write(predictions.reshape(meta['height'], meta['width']), indexes=1)
                dst.write_colormap(1, color_map)


def merge_tiles(path_list, out_path, bounds=None):
    """
    Merge the raster images given in the path list and save the results on disk.

    Parameters
    ----------
    path_list: list
        A list of string storing the path to all the images to merge
    out_path: str
        the path and file name to which the merge is saved
    bounds: tuple
        (left, bottom, right, top) the boundaries to extract from (in UTM).

    Returns
    -------
    None
    """
    # open all tiles
    src_file_mosaic = []
    for fpath in path_list:
        src = rasterio.open(fpath)
        src_file_mosaic.append(src)
    # merge the files into a single mosaic
    mosaic, out_trans = rasterio.merge.merge(src_file_mosaic, bounds=bounds)
    # update
    out_meta = src.meta.copy()
    out_meta.update(
        {'driver': 'GTiff', 'height': mosaic.shape[1], 'width': mosaic.shape[2], 'transform': out_trans})
    # save the merged
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(mosaic)


def mosaic_predictions():
    pass
