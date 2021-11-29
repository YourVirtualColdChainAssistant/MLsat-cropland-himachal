import os
import re
import datetime
import numpy as np
import pandas as pd
from src.utils.util import load_geotiff
from itertools import groupby
import matplotlib.pyplot as plt
import collections


def stack_all_timestamps(logger, from_dir, meta, window=None, way='weekly', interpolation='previous',
                         check_filling=False):
    """
    Stack all the timestamps in from_dir folder, ignoring all black images.

    :param logger: std::out
    :param from_dir: string
    :param window: rasterio.window.Window
    :param way: string
        choices = ['raw', 'weekly', 'monthly']
    :param interpolation: string
        choices = ['zero', 'previous']
    :param check_filling: bool

    :return: bands_array, meta, timestamps_bf, timestamps_af, timestamps_weekly
    bands_array: array
        shape (n_pixels, n_bands, n_weeks)
    timestamps_bf: the raw timestamps
    """

    # ### sanity check
    choices_sanity_check(['raw', 'weekly', 'monthly'], way, 'way')
    choices_sanity_check(['zero', 'previous'], interpolation, 'interpolation')

    # ### raw files
    # sorted available files
    filenames = sorted([file for file in os.listdir(from_dir) if file.endswith('tiff')])

    # find all the raw time stamps
    timestamps_bf = []
    for filename in filenames:
        timestamps_bf.append(datetime.datetime.strptime(re.split('[_.]', filename)[-2], '%Y%m%dT%H%M%S%f').date())

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

    # ### stack all the timestamps
    timestamps_af_pd = pd.Series(timestamps_af)
    n_total = meta['height'] * meta['width']
    bands_list, black_ids, p_cloud_list, p_fill_list = [], [], [], []
    indicator = []
    for i, timestamp in enumerate(timestamps_ref, start=1):
        # get all the indices
        ids = timestamps_af_pd[timestamps_af_pd.eq(timestamp)].index
        band_list = []
        # with non-empty data, check missing data
        if len(ids) != 0:
            cloud_coverage, nodata_mask_list = [], []
            for id in ids:
                # read band
                filename = filenames[id]
                raster_path = from_dir + filename
                band, meta = load_geotiff(raster_path, window, as_float=True)
                # sanity check
                timestamp_sanity_check(timestamp, raster_path)
                # mask cloud
                cloudy_mask = band[-1] != 0
                nodata_mask = band[0] == 0
                nodata_mask_list.append(nodata_mask)
                band = band[:-1]
                for j in range(len(band)):
                    band[j][cloudy_mask & (~nodata_mask)] = 0
                # pixel values check
                if np.array(band).mean() != 0.0:
                    band_list.append(np.stack(band, axis=2).reshape(-1, len(band)))
                    # Cloud mask is originally available at 60m, resampling can cause inconsistency in data,
                    # so we count cloudy_mask[have_data_mask].sum() rather than cloudy_mask.sum().
                    # The latter can cause percentage greater than 1.
                    cloud_coverage.append(round(cloudy_mask[~nodata_mask].sum() / (~nodata_mask).sum(), 4))
                else:
                    black_ids.append(id)
        # TODO: check how to fill missing data in classic ML models
        # stack by index
        if len(band_list) != 0:
            # merge images of a period by taking max
            band_list = np.stack(band_list, axis=2).max(axis=2)
            # check the real number of no data, influenced by merging several images
            nodata_mask = nodata_mask_list[0]
            for m in nodata_mask_list:
                nodata_mask = nodata_mask & m
            n_nodata = nodata_mask.sum()
            # use forward filling for no_data
            zero_mask = band_list[:, 0] == 0
            indicator.append(zero_mask)
            n_zero = zero_mask.sum()
            # forward filling
            if i != 1:
                for k in range(meta['count'] - 1):
                    band_list[zero_mask, k] = bands_list[-1][zero_mask, k]
            p_cloud_list.append(round((n_zero - n_nodata) / n_total, 4))
            p_fill_list.append(round(n_zero / n_total, 4))
        elif interpolation == 'zero' or i == 1:
            band_list = np.zeros((meta['height'] * meta['width'], meta['count'] - 1))
            p_cloud_list.append(0)
            p_fill_list.append(1)
            indicator.append(np.ones((meta['height'] * meta['width'])))
        else:  # interpolation == 'previous'
            band_list = bands_list[-1]
            p_cloud_list.append(p_cloud_list[i - 2])
            p_fill_list.append(1)
            indicator.append(np.ones((meta['height'] * meta['width'])))

        # print
        if len(ids) != 0:
            print_str = ''
            for id in ids:
                if id in black_ids:
                    print_str += f'x{timestamps_bf[id].strftime("%Y-%m-%d")}, '
                else:
                    print_str += f'{timestamps_bf[id].strftime("%Y-%m-%d")}, '
            logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} ({print_str})')
            if len(cloud_coverage):
                logger.info(f'          cloud_coverage {cloud_coverage} => {p_cloud_list[i - 1]:.4f}')
        elif interpolation == 'zero' or i == 1:
            logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} (0)')
        else:
            logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} (previous)')
        logger.info(f'          filling ratio = {p_fill_list[i - 1]}')
        # TODO: cloud and filling ratio is not correctly calculated (include pixels not test)

        bands_list.append(band_list)

    if check_filling:
        # check filling occurrence
        ind = np.array(indicator.copy())
        counts = dict(sorted(collections.Counter(list(ind.sum(axis=0))).items()))
        plt.figure()
        plt.bar(counts.keys(), counts.values())
        plt.xlim(0, len(timestamps_ref))
        plt.xlabel('Filling counts')
        plt.ylabel('Occurrence')
        plt.savefig('../figs/filling_occurrence.png', bbox_inches='tight')
        save_dict_to_df(counts, '../figs/filling_occurrence.csv')

        consecutive_highest = []
        for i in range(n_total):
            _, num = highest_occ(list(ind[:, i]))
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
            num = count_occ(list(ind[:, i]))
            consecutive += num
        consecutive_counts = dict(sorted(collections.Counter(consecutive).items()))
        plt.figure()
        plt.bar(consecutive_counts.keys(), consecutive_counts.values())
        plt.xlim(0, len(timestamps_ref))
        plt.xlabel('Filling counts')
        plt.ylabel('Consecutive occurrence')
        plt.savefig('../figs/filling_consecutive_occurrence.png', bbox_inches='tight')
        save_dict_to_df(consecutive_counts, '../figs/filling_consecutive_occurrence.csv')

    # stack finally
    bands_array = np.stack(bands_list, axis=2)
    meta.update(count=meta['count'] - 1)
    logger.info(f'  avg. cloud coverage = {np.array(p_cloud_list).mean():.4f}')
    logger.info(f'  avg. filling ratio = {np.array(p_fill_list).mean():.4f}')
    logger.info('  ok')

    return bands_array, meta, timestamps_bf, timestamps_af, timestamps_ref


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


def timestamp_sanity_check(timestamp_std, filename):
    timestamp_get = datetime.datetime.strptime(re.split('[_.]', filename)[-2],
                                               '%Y%m%dT%H%M%S%f').date()
    timestamp_get = timestamp_get - datetime.timedelta(days=timestamp_get.weekday())
    if timestamp_std != timestamp_get:
        raise ValueError(f'{timestamp_std} and {timestamp_get} do not match!')


def choices_sanity_check(choices, choice, var_name):
    if choice not in choices:
        raise ValueError(f'{choice} is unavailable. Please choose "{var_name}" from {choices}')


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
