import os
import re
import datetime
import numpy as np
import pandas as pd
from src.utils.util import load_geotiff


def stack_all_timestamps(logger, from_dir, way='weekly', interpolation='previous'):
    """
    Stack all the timestamps in from_dir folder, ignoring all black images.

    :param logger: std::out
    :param from_dir: string
    :param way: string
        choices = ['raw', 'weekly', 'monthly']
    :param interpolation: string
        choices = ['zero', 'previous']

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
    # get bands' meta data
    _, meta = load_geotiff(from_dir + filenames[0], as_float=False)
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
                band, meta = load_geotiff(raster_path, as_float=True)
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
            n_zero = zero_mask.sum()
            # # check whether all bands have NoData in the same position
            # masks = []
            # for k in range(meta['count'] - 1):
            #     masks.append((band_list[:, k] == 0).sum())
            # if not masks[0] == masks[1] == masks[2] == masks[3]:
            #     logger.info(f'         NoData number in four bands are {masks}')
            if i != 1:
                for k in range(meta['count'] - 1):
                    band_list[zero_mask, k] = bands_list[-1][zero_mask, k]
            p_cloud_list.append(round((n_zero - n_nodata) / n_total, 4))
            p_fill_list.append(round(n_zero / n_total, 4))
        elif interpolation == 'zero' or i == 1:
            band_list = np.zeros((meta['height'] * meta['width'], meta['count'] - 1))
            p_cloud_list.append(0)
            p_fill_list.append(1)
        else:  # interpolation == 'previous'
            band_list = bands_list[-1]
            p_cloud_list.append(p_cloud_list[i - 2])
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
        elif interpolation == 'zero' or i == 1:
            logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} (0)')
        else:
            logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} (previous)')
        if len(cloud_coverage):
            logger.info(f'          cloud_coverage {cloud_coverage} => {p_cloud_list[i - 1]:.4f}')
        logger.info(f'          filling ratio = {p_fill_list[i - 1]}')

        bands_list.append(band_list)

    # stack finally
    bands_array = np.stack(bands_list, axis=2)
    meta.update(count=meta['count'] - 1)
    logger.info(f'  avg. cloud coverage = {np.array(p_cloud_list).mean():.4f}')
    logger.info(f'  avg. filling ratio = {np.array(p_fill_list).mean():.4f}')
    logger.info('  ok')

    return bands_array, meta, timestamps_bf, timestamps_af, timestamps_ref


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
