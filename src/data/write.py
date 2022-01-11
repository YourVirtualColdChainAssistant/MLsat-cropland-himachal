import rasterio
from src.evaluation.util import adjust_raster_size


def save_predictions_geotiff(predictions, meta_src, save_path, cat_mask,
                             region_indicator=None, color_by_height=False):
    """
    Save predictions into .tiff file.

    Parameters
    ----------
    predictions: np.array
        shape (height * width, )
    meta_src: dict
        source meta data.
    save_path: str
        path to save .tiff
    cat_mask: np.array
        shape (height * width, )
    region_indicator: str or rasterio.window.Window
        an indicator about the region to save.
    color_by_height: bool
        Whether add a band of mask colored by height.

    Returns
    -------
    None
    """
    color_map = {0: (0, 0, 0), 2: (154, 205, 50)}
    # color_map_2 = {0: (0, 0, 0), 1: (255, 255, 255), 2: (0, 0, 255), 3: (0, 255, 0)}
    # color_map_3 = {-9999: (0,0,0), -9998: (0, 0, 0), 0: (255, 255, 255), 3000: (255, 0, 0)}
    out_meta = meta_src.copy()
    if color_by_height:
        # relative path of altitude data
        height_map_path = '../../../Data/layers_india/ancilliary_data/elevation/IND_alt.vrt'
        adjust_raster_size(height_map_path, './data/open_datasets/height_map.tiff',
                           region_indicator=region_indicator, meta=meta_src, label_only=False)
        with rasterio.open('./data/open_datasets/height_map.tiff') as f:
            height_map = f.read(1).reshape(-1)
        height_map[predictions != 2] = -9998
        with rasterio.Env():
            # Write an array as a raster band to a new 8-bit file. We start with the profile of the source
            out_meta.update(dtype=rasterio.int16, count=3)
            with rasterio.open(save_path, 'w', **out_meta) as dst:
                # reshape into (band, height, width)
                dst.write_band(1, predictions.reshape(out_meta['height'], out_meta['width']).astype(rasterio.int16))
                dst.write_band(2, cat_mask.reshape(out_meta['height'], out_meta['width']).astype(rasterio.int16))
                dst.write_band(3, height_map.reshape(out_meta['height'], out_meta['width']).astype(rasterio.int16))
                dst.write_colormap(1, color_map)
    else:
        with rasterio.Env():
            # Write an array as a raster band to a new 8-bit file. We start with the profile of the source
            out_meta.update(dtype=rasterio.uint8, count=2)
            with rasterio.open(save_path, 'w', **out_meta) as dst:
                # reshape into (band, height, width)
                dst.write(predictions.reshape(out_meta['height'], out_meta['width']), 1)
                dst.write(cat_mask.reshape(out_meta['height'], out_meta['width']), 2)
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
