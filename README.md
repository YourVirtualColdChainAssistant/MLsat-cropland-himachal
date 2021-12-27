# ML4Satellite

### Build environment

Wheels can be downloaded from https://www.lfd.uci.edu/~gohlke/pythonlibs/.

```buildoutcfg
conda create -n sat python=3.9
conda activate sat

# run as administrator to install from wheels
pip install Fiona-1.8.20-cp39-cp39-win_amd64.whl GDAL-3.3.2-cp39-cp39-win_amd64.whl pyproj-3.2.1-cp39-cp39-win_amd64.whl rasterio-1.2.8-cp39-cp39-win_amd64.whl Shapely-1.8.0-cp39-cp39-win_amd64.whl
# if there is "AttributeError: module 'brotli' has no attribute 'error'",  
# run `conda install -c anaconda urllib3` first 
 
pip install spacv sentinelsat pulearn rtree 

conda install -c conda-forge scikit-image scikit-gstat
```

Install in Linux:

```buildoutcfg
conda install -c conda-forge scikit-image scikit-gstat gdal 

pip install spacv sentinelsat pulearn rtree fiona pyproj rasterio shapely 
```