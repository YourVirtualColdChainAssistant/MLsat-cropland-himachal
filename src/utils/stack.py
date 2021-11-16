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
    _, meta = load_geotiff(from_dir + filenames[0])
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
    bands_list, black_ids, cloud_coverage_list = [], [], []
    for i, timestamp in enumerate(timestamps_ref, start=1):
        # get all the indices
        ids = timestamps_af_pd[timestamps_af_pd.eq(timestamp)].index
        band_list = []
        # with non-empty data, check missing data
        if len(ids) != 0:
            cloud_coverage = []
            for id in ids:
                # read band
                filename = filenames[id]
                raster_path = from_dir + filename
                band, meta = load_geotiff(raster_path)
                # sanity check
                timestamp_sanity_check(timestamp, raster_path)
                # mask cloud
                cloudy_mask = band[-1] != 0
                have_data_mask = band[0] != 0
                for j, b in enumerate(band[:-1]):
                    band[j][cloudy_mask] = 0
                band = band[:-1]
                # pixel values check
                if np.array(band).mean() != 0.0:
                    band_list.append(np.stack(band, axis=2).reshape(-1, len(band)))
                    cloud_coverage.append(round(cloudy_mask.sum() / have_data_mask.sum(), 2))
                else:
                    black_ids.append(id)

        # stack by index
        if len(band_list) != 0 and i != 1:
            band_list = np.stack(band_list, axis=2).max(axis=2)  # take max
            # use forward filling for no_data
            nodata_mask = band_list[:, 0] == 0
            for k in range(meta['count'] - 1):
                band_list[nodata_mask, k] = bands_list[-1][nodata_mask, k]
            cloud_coverage_list.append(np.array(cloud_coverage).mean())
        elif interpolation == 'zero' or i == 1:
            band_list = np.zeros((meta['height'] * meta['width'], meta['count'] - 1))
        else:  # interpolation == 'previous'
            band_list = bands_list[-1]

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
                logger.info(f'         cloud_coverage {cloud_coverage} => {np.array(cloud_coverage).mean():.2f}')
        elif interpolation == 'zero' or i == 1:
            logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} (0)')
        else:
            logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} (previous)')

        bands_list.append(band_list)

    # stack finally
    bands_array = np.stack(bands_list, axis=2)
    logger.info(f'  avg. cloud coverage = {np.array(cloud_coverage_list).mean():.2f}')
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
