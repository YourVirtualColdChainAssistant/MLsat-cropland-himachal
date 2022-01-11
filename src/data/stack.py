import os
import re
import datetime
import numpy as np
import pandas as pd
import collections
from itertools import groupby
import matplotlib.pyplot as plt

from src.data.load import load_geotiff


def stack_timestamps(logger, from_dir, meta, descriptions, window=None, read_as='as_integer',
                     way='weekly', check_missing=False):
    """
    Stack all the timestamps in from_dir folder, ignoring all black images.

    Legend for scene classification, corresponding to 0 to 11:
        ['NO_DATA', 'SATURATED_OR_DEFECTIVE', 'DARK_AREA_PIXELS', 'CLOUD_SHADOWS', 'VEGETATION', 'NOT_VEGETATED',
         'WATER', 'UNCLASSIFIED', 'CLOUD_MEDIUM_PROBABILITY', 'CLOUD_HIGH_PROBABILITY', 'THIN_CIRRUS', 'SNOW']

    Parameters
    ----------
    logger
    from_dir: str
        path storing satellite raster images.
    meta: dict
        Meta data
    descriptions: list
        A list of band name.
    window: rasterio.window.Window or None
        The window to load images.
    read_as: choices = [as_integer, as_float, as_reflectance]
    way: [raw, weekly, monthly]
    check_missing: bool

    Returns
    -------
    bands_array: np.array
        shape (height, width, n_bands, n_weeks)
    cat_pixel: np.array
        shape (height * width, )
    timestamps_bf: the raw timestamps
    """
    if way not in ['raw', 'weekly', 'monthly']:
        raise ValueError(f"{way} is unavailable. Choose from ['raw', 'weekly', 'monthly']")
    filenames = sorted([file for file in os.listdir(from_dir) if file.endswith('tiff')])
    timestamps_bf, timestamps_af, timestamps_ref = get_timestamps(filenames, way)

    # stack all the timestamps
    timestamps_af_pd = pd.Series(timestamps_af)
    n_total = meta['height'] * meta['width']
    idx_cloud = descriptions.index('cloud mask')
    idx_other = list(range(meta['count']))
    idx_other.pop(idx_cloud)
    meta.update({'count': len(idx_other)})
    band_lists, cat_mask_lists, black_ids, p_cloud_list, p_fill_list, p_zero_list = [], [], [], [], [], []
    for i, timestamp in enumerate(timestamps_ref, start=1):
        # get all the indices
        ids = timestamps_af_pd[timestamps_af_pd.eq(timestamp)].index
        band_list, cat_mask_list = [], []
        # with non-empty data, check missing data
        if len(ids) != 0:
            for id in ids:
                # read band
                raster_path = from_dir + filenames[id]
                print('Before loading data:', datetime.datetime.now())
                band, _ = load_geotiff(raster_path, window, read_as)
                print('After loading data:', datetime.datetime.now())
                cat_mask = categorize_scene_classification(band[idx_cloud])
                band = np.stack(band, axis=2)[:, :, idx_other]
                # pixel values check
                if (cat_mask == 0).sum() != n_total:
                    band_list.append(band.reshape(-1, meta['count']))
                    cat_mask_list.append(cat_mask.reshape(-1))
                else:  # all empty 
                    black_ids.append(id)

        # stack by index
        if len(band_list) != 0:
            # merge images of a period by taking max
            band_list, cat_mask_list = get_cloudless(band_list, cat_mask_list)
            # check gaps
            n_zero = (band_list[:, 0] == 0).sum()
            n_nodata = (cat_mask_list == 0).sum()
            n_cloudy = (cat_mask_list == 1).sum()
            p_zero_list.append(round(n_zero / n_total, 4))
            p_cloud_list.append(round(n_cloudy / n_total, 4))
            p_fill_list.append(round((n_nodata + n_cloudy) / n_total, 4))
        else:  # i == 1:
            band_list = np.zeros((meta['height'] * meta['width'], meta['count']))
            cat_mask_list = np.zeros(meta['height'] * meta['width'])
            p_cloud_list.append(1)
            p_fill_list.append(1)

        # print
        if len(ids) != 0:
            print_str = ''
            for id in ids:
                if id in black_ids:
                    print_str += f'x{timestamps_bf[id].strftime("%Y-%m-%d")}, '
                else:
                    print_str += f'{timestamps_bf[id].strftime("%Y-%m-%d")}, '
            logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} ({print_str})')
        else:  # i == 1 or
            logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} (0)')
        band_lists.append(band_list)
        cat_mask_lists.append(cat_mask_list)

    logger.info(f'  avg. cloud coverage = {np.array(p_cloud_list).mean():.4f}')
    logger.info(f'  avg. filling ratio = {np.array(p_fill_list).mean():.4f}')
    logger.info(f'  avg. zero ratio = {np.array(p_zero_list).mean():.4f}')

    # decide cat for each pixel 
    cat_pixel = categorize_pixel(cat_mask_lists)

    # stack finally
    bands_array = np.stack(band_lists, axis=2).reshape(meta['height'], meta['width'], meta['count'], -1)

    return bands_array, cat_pixel, meta, timestamps_bf, timestamps_ref


def get_timestamps(filenames, way):
    # find all the raw time stamps
    timestamps_bf = [datetime.datetime.strptime(re.split('[_.]', filename)[-2], '%Y%m%dT%H%M%S%f').date() for filename
                     in filenames]

    # ### check the way to stack
    if way == 'raw':
        timestamps_af = timestamps_bf
        timestamps_ref = timestamps_bf
    elif way == 'weekly':
        timestamps_af = [ts - datetime.timedelta(days=ts.weekday()) for ts in
                         timestamps_bf]  # datetime.weekday() returns 0~6
        timestamps_ref = get_weekly_timestamps()
    else:
        timestamps_af = [ts - datetime.timedelta(days=ts.day - 1) for ts in timestamps_bf]  # datetime.day returns 1-31
        timestamps_ref = get_monthly_timestamps()

    return timestamps_bf, timestamps_af, timestamps_ref


def get_weekly_timestamps():
    """
    Get the date of all Monday in 2020.

    """
    date_start = datetime.date(2020, 1, 1)
    date_end = datetime.date(2020, 12, 31)
    date_start = date_start - datetime.timedelta(days=date_start.weekday())
    date_end = date_end - datetime.timedelta(days=date_end.weekday())
    d = date_start
    weekly_timestamps = []
    while d <= date_end:
        weekly_timestamps.append(d)
        d += datetime.timedelta(7)
    return weekly_timestamps


def get_monthly_timestamps():
    """
    Get the first day of each month in 2020.

    """
    date_start = datetime.date(2020, 1, 1)
    monthly_timestamps = []
    for m in range(1, 13):
        monthly_timestamps.append(date_start.replace(month=m))
    return monthly_timestamps


def categorize_scene_classification(scene):
    """
    Categorize original 11 scenes into 4 groups.
    0 = nodata,
    1 = cloudy,
    2 = no need to predict,
    3 = normal

    Parameters
    ----------
    scene: np.array
        shape (height, width)

    Returns
    -------
    cat_mask: np.array
        shape (height, width)
    """
    cat_mask = np.zeros_like(scene)
    cat_mask[(scene == 1) | (scene == 2) | (scene == 4) | (scene == 5)] = 3
    cat_mask[(scene == 6) | (scene == 11)] = 2
    cat_mask[(scene == 3) | (scene == 7) | (scene == 8) | (scene == 9) | (scene == 10)] = 1
    return cat_mask


def get_cloudless(band_list, cat_mask_list):
    """
    Get cloudless bands from possibly multiple bands.

    Parameters
    ----------
    band_list: A list of nd.array
        shape (height * width, n_band)
    cat_mask_list: A list of nd.array
        shape (height * width, )

    Returns
    ----------
    band_cloudless: np.array
        shape ()
    cat_mask_cloudless: np.array
        shape (height * width, )
    """
    band_stacked = np.stack(band_list, axis=2)
    cat_mask_stacked = np.stack(cat_mask_list, axis=1)
    cat_mask_cloudless = cat_mask_stacked.max(axis=1)
    # get cloudless by taking max
    cloudy_mask = (cat_mask_cloudless == 1)
    argmax = cat_mask_stacked.argmax(axis=1)
    band_cloudless = band_stacked[np.arange(argmax.shape[0]), :, argmax]
    # fill cloudy data as 0 for further processing
    band_cloudless[cloudy_mask] = 0
    return band_cloudless, cat_mask_cloudless


def categorize_pixel(cat_mask_lists):
    """
    Cetagorize each pixel given the time series mask.

    Parameters
    ----------
    cat_mask_lists: A list of nd.array
        n_col of shape (height * width)

    Returns
    ----------
    cat_pixel: np.array
        shape (n_data, )
    """
    cat_mask_stacked = np.stack(cat_mask_lists, axis=1)
    n_data, n_col = cat_mask_stacked.shape
    n0 = (cat_mask_stacked == 0).sum(axis=1)
    n1 = (cat_mask_stacked == 1).sum(axis=1)
    n2 = (cat_mask_stacked == 2).sum(axis=1)
    n3 = (cat_mask_stacked == 3).sum(axis=1)

    cat_pixel = np.zeros(n_data)  # by default = No Data
    more = np.stack([n2, n3], axis=1).argmax(axis=1)
    cat_pixel[((n0 + n1) != n_col) & (more == 0)] = 2  # don't predict
    cat_pixel[((n0 + n1) != n_col) & (more == 1)] = 3  # predict

    return cat_pixel


def check_missing_condition(zero_mask_list, timestamps_ref, n_total):
    # check filling occurrence
    zero_mask_arr = np.array(zero_mask_list)
    counts = dict(sorted(collections.Counter(list(zero_mask_arr.sum(axis=0))).items()))
    plt.figure()
    plt.bar(counts.keys(), counts.values())
    plt.xlim(0, len(timestamps_ref))
    plt.xlabel('Filling counts')
    plt.ylabel('Occurrence')
    plt.savefig('./figs/filling_occurrence.png', bbox_inches='tight')
    save_dict_to_df(counts, './figs/filling_occurrence.csv')

    consecutive_highest = []
    for i in range(n_total):
        _, num = highest_occ(list(zero_mask_arr[:, i]))
        consecutive_highest.append(num)
    consecutive_counts_highest = dict(sorted(collections.Counter(consecutive_highest).items()))
    plt.figure()
    plt.bar(consecutive_counts_highest.keys(), consecutive_counts_highest.values())
    plt.xlim(0, len(timestamps_ref))
    plt.xlabel('Filling counts')
    plt.ylabel('Highest consecutive occurrence')
    plt.savefig('./figs/filling_highest_consecutive_occurrence.png', bbox_inches='tight')
    save_dict_to_df(consecutive_counts_highest, './figs/filling_highest_consecutive_occurrence.csv')

    consecutive = []
    for i in range(n_total):
        num = count_occ(list(zero_mask_arr[:, i]))
        consecutive += num
    consecutive_counts = dict(sorted(collections.Counter(consecutive).items()))
    plt.figure()
    plt.bar(consecutive_counts.keys(), consecutive_counts.values())
    plt.xlim(0, len(timestamps_ref))
    plt.xlabel('Filling counts')
    plt.ylabel('Consecutive occurrence')
    plt.savefig('./figs/filling_consecutive_occurrence.png', bbox_inches='tight')
    save_dict_to_df(consecutive_counts, './figs/filling_consecutive_occurrence.csv')


def highest_occ(b):
    occurrence, num_times = 0, 0
    for key, values in groupby(b, lambda x: x):
        if key == 1:
            val = len(list(values))
            if val >= num_times:
                occurrence, num_times = key, val
    return occurrence, num_times


def count_occ(b):
    num = []
    for key, values in groupby(b, lambda x: x):
        if key == 1:
            num.append(len(list(values)))
    return num


def save_dict_to_df(d, save_path):
    df = pd.DataFrame()
    df['con'] = d.keys()
    df['occ'] = d.values()
    df.to_csv(save_path, index=False)
