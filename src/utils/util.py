import fiona
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import skimage
import skimage.draw
import pyproj
import rasterio


def load_geotiff(path, window=None):
    """ Load the geotiff as a list of numpy array.
        INPUT : path (str) -> the path to the geotiff
                window (rasterio.windows.Window) -> the window to use when loading the image
        OUTPUT : band (list of numpy array) -> the different bands unscaled
                 meta (dictionary) -> the metadata associated with the geotiff
    """
    with rasterio.open(path) as f:
        band = [skimage.img_as_uint(f.read(i + 1, window=window)) for i in range(f.count)]
        meta = f.meta
        if window is not None:
            meta['height'] = window.height
            meta['width'] = window.width
            meta['transform'] = f.window_transform(window)
    return band, meta


def save_predictions_geotiff(meta_src, predictions, save_path):
    # Register GDAL format drivers and configuration options with a context manager
    color_map = {
        0: (0, 0, 0),
        1: (240, 65, 53),
        2: (154, 205, 50),
        3: (184, 134, 11),
        255: (255, 255, 255)
    }
    with rasterio.Env():
        # Write an array as a raster band to a new 8-bit file. We start with the profile of the source
        out_meta = meta_src.copy()
        out_meta.update(
            dtype=rasterio.uint8,
            count=1)
        with rasterio.open(save_path, 'w', **out_meta) as dst:
            # reshape into (band, height, width)
            dst.write(predictions.reshape(1, out_meta['height'], out_meta['width']).astype(rasterio.uint8))
            dst.write_colormap(1, color_map)


def load_target_shp(path, transform=None, proj_out=None):
    """ Load the shapefile as a list of numpy array of coordinates
        INPUT : path (str) -> the path to the shapefile
                transform (rasterio.Affine) -> the affine transformation to get the polygon in row;col format from UTM.
        OUTPUT : poly (list of np.array) -> list of polygons (as numpy.array of coordinates)
                 poly_rc (list of np.array) -> list of polygon in row-col format if a transform is given
    """
    print("Loading target shapefile...")
    with fiona.open(path) as shapefile:
        proj_in = pyproj.Proj(shapefile.crs)
        class_type = [feature['properties']['id'] for feature in shapefile]
        features = [feature["geometry"] for feature in shapefile]
    # re-project polygons if necessary
    if proj_out is None or proj_in == proj_out:
        poly = [np.array([(coord[0], coord[1]) for coord in features[i]['coordinates'][0]]) for i in
                range(len(features))]
        print('No re-projection!')
    else:
        poly = [np.array(
            [pyproj.transform(proj_in, proj_out, coord[0], coord[1]) for coord in features[i]['coordinates'][0]]) for i
            in range(len(features))]
        print(f'Re-project from {proj_in} to {proj_out}')

    poly_rc = None
    # transform in row-col if a transform is given
    if transform is not None:
        poly_rc = [np.array([rasterio.transform.rowcol(transform, coord[0], coord[1])[::-1] for coord in p]) for p in
                   poly]
    print('Loaded target shape files.')

    return features, poly, poly_rc, class_type


def compute_mask(polygon_list, meta, val_list):
    """ Get mask of class of a polygon list
        INPUT : polygon_list (list od polygon in coordinates (x, y)) -> the polygons in row;col format
                meta -> the image width and height
                val_list(list of int) -> the class associated with each polygon
        OUTPUT : img (np.array 2D) -> the mask in which the pixel value reflect it's class (zero being the absence of class)
    """
    img = np.zeros((meta['height'], meta['width']), dtype=np.uint8)  # skimage : row,col --> h,w
    i = 0
    for polygon, val in zip(polygon_list, val_list):
        rr, cc = skimage.draw.polygon(polygon[:, 1], polygon[:, 0], img.shape)
        img[rr, cc] = val
        i += 1
    print("Added targets' mask.")
    return img


def convert_gdf_to_shp(data, save_path):
    with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
        data.to_file(save_path)


def read_shp(file_path):
    with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
        data = gpd.read_file(file_path)
    return data


def count_classes(logger, y):
    tot_num = len(y)
    for i in np.unique(y):
        y_i = y[y == i]
        msg = f'  label = {i}, pixel number = {y_i.shape[0]}, percentage = {round(len(y_i) / tot_num * 100, 2)}%'
        if logger is None:
            print(msg)
        else:
            logger.info(msg)


def dropna_in_shapefile(from_shp_path, to_shp_path=None):
    shapefile = gpd.read_file(from_shp_path)
    shapefile = shapefile.dropna().reset_index(drop=True)
    if to_shp_path is None:
        shapefile.to_file(from_shp_path)
    else:
        shapefile.to_file(to_shp_path)


def multipolygons_to_polygons(shp_file):
    # check the number of multipolygons
    multi_polygons_df = shp_file[shp_file['geometry'].type == 'MultiPolygon']
    polygons_df = shp_file[shp_file['geometry'].type == 'Polygon']
    if multi_polygons_df.shape[0] == 0:
        print('No multi-polygons!')
    else:
        new_polygons = []
        num_multi_polygons = multi_polygons_df.shape[0]
        print(f'Converting {num_multi_polygons} multi-polygons to polygons...')
        for i in range(num_multi_polygons):
            multi_polygon_ = multi_polygons_df.iloc[i]
            label, district, multi_polygon = multi_polygon_.id, multi_polygon_.district, multi_polygon_.geometry
            for polygon in multi_polygon.geoms:
                new_polygons.append([label, district, polygon])
        new_polygons_df = pd.DataFrame(new_polygons, columns=['id', 'district', 'geometry'])
        polygons_df = pd.concat([polygons_df, new_polygons_df], axis=0)
    return polygons_df


def correct_predictions():
    pass


def resample_negatives(pos, neg):
    pass


def get_cropland_mask(df, n_feature, pretrained_name):
    # read pretrained model
    trained_model = pickle.load(open(f'../models/{pretrained_name}.sav', 'rb'))

    # new columns
    df['gt_cropland'] = 0
    df.loc[(df.label.values == 1) | (df.label.values == 2), 'gt_cropland'] = 1

    # cropland
    to_pred_mask = df.label.values == 0
    preds = trained_model.predict(df.iloc[to_pred_mask, :n_feature].values)
    cropland_pred = np.empty_like(preds)
    cropland_pred[preds == 2] = 1
    cropland_pred[preds == 3] = 0
    df['gp_cropland'] = df['gt_cropland'].copy()
    df.loc[to_pred_mask, 'gp_cropland'] = cropland_pred
    df['gp_cropland_mask'] = False
    df.loc[df.gp_cropland.values == 1, 'gp_cropland_mask'] = True

    return df.gp_cropland_mask.values


def save_cv_results(res, save_path):
    df = pd.DataFrame()
    df['mean_fit_time'] = res['mean_fit_time']
    df['std_fit_time'] = res['std_fit_time']
    df['mean_score_time'] = res['mean_score_time']
    df['std_score_time'] = res['std_score_time']
    df['params'] = res['params']
    df['mean_test_score'] = res['mean_test_score']
    df['std_test_score'] = res['std_test_score']
    df['rank_test_score'] = res['rank_test_score']
    df.to_csv(save_path, index=False)


def merge_tiles(path_list, out_path, bounds=None):
    """ Merge the raster images given in the path list and save the results on disk.
        INPUT : path_list (list of string) -> the path to all the images to merge
                out_path (str) -> the path and file name to which the merge is saved
                bounds (tuple) -> (left, bottom, right, top) the boundaries to extract from (in UTM).
        OUTPUT : None
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
