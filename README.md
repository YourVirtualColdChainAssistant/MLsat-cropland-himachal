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

## Repository Structure

The following is a description of the structure of this template repository.

```buildoutcfg
    .
    ├── README.md                   > This file contains a description of what is the repository used for and how to use it.
    ├── LOG.xlsx                    > File for keeping track of important experiments and milestones. 
    ├── requirements.txt            > File containing the packages and their versions.
    ├── .gitignore                  > File for telling git to not track changes on.
    ├── data                        > Folder with ground truths and ancillary data.
    |    ├── config                 > Folder to hold configuration to use in train and prediction.
    |    ├── ground_truth           > Folder to hold ground truth data.
    |    └── open_datasets          > Folder to hold open datasets.
    ├── figs                        > Folder for keeping the media files in this project. 
    ├── logs                        > Folder for keeping the log data of training and predicting.      
    ├── models                      > Folder for keeping useful trained models. This folder is to .gitignore.
    ├── notebooks                   > Folder for keeping all notebooks, used mainly for exploratory data analysis.              
    └── src                         > Folder for keeping the source code of this project.        
        ├── data                    > Folder containing all .py files used for data preparation. 
        |    ├── __init__.py        > Makes this folder an importable python module.
        |    ├── download.py        > Script for downloading raw data to raw folder in N drive.
        |    ├── preprocess.py      > Script for preprocessing raw data (include unzip, atmopheric correction, merge). 
        |    ├── load.py            > Script for loading images and clean ground truth data. 
        |    ├── write.py           > Script for writing predictions or post-processing. 
        |    ├── prepare.py         > Script for preparing images for modeling. 
        |    ├── engineer.py        > Script for engineering features. 
        |    ├── clip.py            > Script for clip images. 
        |    └── stack.py           > Script for stacking all time series. 
        |
        ├── models                  > Folder containing all .py with models. 
        |    ├── __init__.py        > Makes this folder an importable python module.
        |    ├── cropland.py        > Script for cropland classification. 
        |    ├── crop_type.py       > Script for crop type classification. 
        |    └── util.py            > Script generally used in both classification. 
        |    
        ├── evaluation              > Folder containing all .py files used for evaluation. 
        |    ├── __init__.py        > Makes this folder an importable python module.
        |    ├── evaluate.py        > Script for evaluating models. 
        |    └── visualize.py       > Scripts to create figures. Save to figures folder.
        |    
        ├── utils                   > Folder contraining all util functionality. 
        |    ├── __init__.py        > Makes this folder an importable python module.
        |    ├── util.py            > Script of general util functions.
        |    ├── logger.py          > Script for generating log file. 
        |    └── scv.py             > Script for spatial cross validation. 
        |
        ├── __init__.py             > Makes this folder an importable python module.
        ├── train_cropland.py       > Script to train cropland model.  
        ├── predict_cropland.py     > Script to predict cropland masks given pre-trained model.  
        ├── train_crop_type.py      > Script to train corp type model.  
        └── predict_crop_type.py    > Script to predict corp type masks given pre-trained model.  
```
