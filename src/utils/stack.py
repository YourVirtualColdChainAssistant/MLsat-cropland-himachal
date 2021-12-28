import os
import re
import datetime
import numpy as np
import pandas as pd
from src.utils.util import load_geotiff
from itertools import groupby
import matplotlib.pyplot as plt
import collections


def stack_timestamps(logger, from_dir, meta, descriptions, window=None, read_as='as_raw',
                     way='weekly', check_missing=False):
    """
    Stack all the timestamps in from_dir folder, ignoring all black images.

    Legend for scene classification, corresponding to 0 to 11:
        ['NO_DATA', 'SATURATED_OR_DEFECTIVE', 'DARK_AREA_PIXELS', 'CLOUD_SHADOWS', 'VEGETATION', 'NOT_VEGETATED',
         'WATER', 'UNCLASSIFIED', 'CLOUD_MEDIUM_PROBABILITY', 'CLOUD_HIGH_PROBABILITY', 'THIN_CIRRUS', 'SNOW']

    :param logger: std::out
    :param from_dir: string
    :param meta:
    :params descriptions:
    :param window: rasterio.window.Window
    :param read_as: string
        choices = ['as_raw', 'as_float', 'as_TOA']
    :param way: string
        choices = ['raw', 'weekly', 'monthly']
    :param check_missing: bool

    :return: bands_array, meta, timestamps_bf, timestamps_weekly
    bands_array: array
        shape (height, width, n_bands + 1, n_weeks)
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
    bands_list, black_ids, p_cloud_list, p_fill_list, zero_mask_list = [], [], [], [], []
    for i, timestamp in enumerate(timestamps_ref, start=1):
        # get all the indices
        ids = timestamps_af_pd[timestamps_af_pd.eq(timestamp)].index
        band_list = []
        # with non-empty data, check missing data
        if len(ids) != 0:
            cloud_coverage, nodata_mask_list = [], []
            for id in ids:
                # read band
                raster_path = from_dir + filenames[id]
                print('Before loading data:', datetime.datetime.now())
                band, meta = load_geotiff(raster_path, window, read_as)
                print('After loading data:', datetime.datetime.now())
                # mask cloud, where 0 = nodata, 1 = normal, 2 = no need to predict, 3 = cloud,
                cloud_band = band[idx_cloud]
                cloud_band[(cloud_band == 1) | (cloud_band == 2) | (cloud_band == 4) | (cloud_band == 5)] = 1
                cloud_band[(cloud_band == 3) | (cloud_band <= 10) | (cloud_band >= 8)] = 3
                cloud_band[(cloud_band == 6) | (cloud_band == 11)] = 2
                band[idx_cloud] = cloud_band
                cloudy_mask = cloud_band == 2
                nodata_mask = cloud_band == 0
                # fill pixels with clouds as 0
                for j in idx_other:
                    band[j][cloudy_mask & (~nodata_mask)] = 0
                # pixel values check
                if nodata_mask.sum() == n_total:
                    band_list.append(np.stack(band, axis=2).reshape(-1, len(band)))
                else:
                    black_ids.append(id)

        # stack by index
        if len(band_list) != 0:
            # merge images of a period by taking max
            band_list = np.stack(band_list, axis=2).max(axis=2)
            # check the real number of no data, influenced by merging several images
            # nodata_mask = nodata_mask_list[0]
            # for m in nodata_mask_list:
            #     nodata_mask = nodata_mask & m
            # zero_mask = band_list[:, 0] == 0
            # n_nodata = nodata_mask.sum()
            # n_zero = zero_mask.sum()
            # zero_mask_list.append(zero_mask)
            # p_cloud_list.append(round((n_zero - n_nodata) / n_total, 4))
            # p_fill_list.append(round(n_zero / n_total, 4))
        else:  # i == 1:
            band_list = np.zeros((meta['height'] * meta['width'], meta['count'] - 1))
            # zero_mask_list.append(np.ones((meta['height'] * meta['width'])))
            # p_cloud_list.append(0)
            # p_fill_list.append(1)

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
        # TODO: cloud and filling ratio is not correctly calculated (include pixels not test)
        bands_list.append(band_list)

    # print('Before checking missing:', datetime.datetime.now())
    # if check_missing:
    #     check_missing_condition(zero_mask_list, timestamps_ref, n_total)
    # print('After checking missing:', datetime.datetime.now())
    # stack finally
    # logger.info(f'  avg. cloud coverage = {np.array(p_cloud_list).mean():.4f}')
    # logger.info(f'  avg. filling ratio = {np.array(p_fill_list).mean():.4f}')

    bands_array = np.stack(bands_list, axis=2).reshape(meta['height'], meta['width'], meta['count'], -1)

    return bands_array, meta, timestamps_bf, timestamps_ref


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


def check_missing_condition(zero_mask_list, timestamps_ref, n_total):
    # check filling occurrence
    zero_mask_arr = np.array(zero_mask_list)
    counts = dict(sorted(collections.Counter(list(zero_mask_arr.sum(axis=0))).items()))
    plt.figure()
    plt.bar(counts.keys(), counts.values())
    plt.xlim(0, len(timestamps_ref))
    plt.xlabel('Filling counts')
    plt.ylabel('Occurrence')
    plt.savefig('../figs/filling_occurrence.png', bbox_inches='tight')
    save_dict_to_df(counts, '../figs/filling_occurrence.csv')

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
    plt.savefig('../figs/filling_highest_consecutive_occurrence.png', bbox_inches='tight')
    save_dict_to_df(consecutive_counts_highest, '../figs/filling_highest_consecutive_occurrence.csv')

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
    plt.savefig('../figs/filling_consecutive_occurrence.png', bbox_inches='tight')
    save_dict_to_df(consecutive_counts, '../figs/filling_consecutive_occurrence.csv')


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
