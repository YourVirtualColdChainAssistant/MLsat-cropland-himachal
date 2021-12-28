# ML4Satellite

## Requirements

Firstly, download `Sen2Cor v2.9` stand-alone in your local follows
the [ESA instruction](http://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor-v2-9/). You may
change `sen2cor_path` accordingly in `process.py`.

Secondly, build conda environment. Wheels can be downloaded
from [this website](https://www.lfd.uci.edu/~gohlke/pythonlibs/).

In Windows:

```buildoutcfg
conda create -n sat python=3.9
conda activate sat

# cd to the direct to install from wheels storing the wheels 
# run as administrator
pip install Fiona-1.8.20-cp39-cp39-win_amd64.whl GDAL-3.3.2-cp39-cp39-win_amd64.whl pyproj-3.2.1-cp39-cp39-win_amd64.whl rasterio-1.2.8-cp39-cp39-win_amd64.whl Shapely-1.8.0-cp39-cp39-win_amd64.whl
# if there is "AttributeError: module 'brotli' has no attribute 'error'",  
# run `conda install -c anaconda urllib3` first 
 
pip install spacv sentinelsat pulearn rtree 
conda install -c conda-forge scikit-image scikit-gstat
```

In Linux:

```buildoutcfg
conda install -c conda-forge scikit-image scikit-gstat gdal 
pip install spacv sentinelsat pulearn rtree fiona pyproj rasterio shapely 
```

## Data Preparation

To download sentinel-2 imagery by tile and year :

```buildoutcfg
python -m src.data.download --tile_id 43SFR --year 2020 --user usr --password pwd
```

To preprocess satellite images, including unzip raw data, atmospheric correction, and stack temporal data (one can pass
several tile ids in one time):

```buildoutcfg
python -m src.data.process --tile_ids 43SFR 43RGQ ...
```

## Train

To train cropland models (with your preference):

```buildoutcfg
python -m src.train_cropland [--arguments]
```

To predict cropland masks (with exactly the same argument settings found in trainining):

```buildoutcfg
python -m src.predict_cropland [--arguments]
```