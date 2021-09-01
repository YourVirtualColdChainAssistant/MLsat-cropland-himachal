import fiona
import numpy as np
import geopandas as gpd
import skimage.draw
import pyproj
import rasterio
from rasterio.mask import mask


def dropna_in_shapefile(from_shp_path, to_shp_path=None):
    shapefile = gpd.read_file(from_shp_path)
    shapefile = shapefile.dropna().reset_index(drop=True)
    if to_shp_path is None:
        shapefile.to_file(from_shp_path)
    else:
        shapefile.to_file(to_shp_path)


def load_target_shp(path, transform=None, proj_out=None):
    """ Load the shapefile as a list of numpy array of coordinates
        INPUT : path (str) -> the path to the shapefile
                transform (rasterio.Affine) -> the affine transformation to get the polygon in row;col format from UTM.
        OUTPUT : poly (list of np.array) -> list of polygons (as numpy.array of coordinates)
                 poly_rc (list of np.array) -> list of polygon in row-col format if a transform is given
    """
    with fiona.open(path) as shapefile:
        proj_in = pyproj.Proj(shapefile.crs)
        class_type = [feature['properties']['id'] for feature in shapefile]
        features = [feature["geometry"] for feature in shapefile]
    # reproject polygons if necessary
    if proj_out is None or proj_in == proj_out:
        poly = [np.array([(coord[0], coord[1]) for coord in features[i]['coordinates'][0]]) for i in
                range(len(features))]
        print('No reprojection!')
    else:
        poly = [np.array(
            [pyproj.transform(proj_in, proj_out, coord[0], coord[1]) for coord in features[i]['coordinates'][0]]) for i
                in range(len(features))]
        print(f'Reproject from {proj_in} to {proj_out}')

    poly_rc = None
    # transform in row-col if a transform is given
    if not transform is None:
        poly_rc = [np.array([rasterio.transform.rowcol(transform, coord[0], coord[1])[::-1] for coord in p]) for p in
                   poly]

    return poly, poly_rc, class_type


def compute_mask(polygon_list, img_w, img_h, val_list):
    """ Get mask of class of a polygon list
        INPUT : polygon_list (list od polygon in coordinates (x, y)) -> the polygons in row;col format
                img_w (int) -> the image width
                img_h (int) -> the image height
                val_list(list of int) -> the class associated with each polygon
        OUTPUT : img (np.array 2D) -> the mask in which the pixel value reflect it's class (zero being the absence of class)
    """
    img = np.zeros((img_h, img_w), dtype=np.uint8)  # skimage : row,col --> h,w
    for polygon, val in zip(polygon_list, val_list):
        rr, cc = skimage.draw.polygon(polygon[:, 1], polygon[:, 0], img.shape)
        img[rr, cc] = val

    return img