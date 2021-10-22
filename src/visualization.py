import os
import re
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import rasterio
import pyproj
import pandas as pd
import datetime
from feature_engineering import add_bands
from util import get_weekly_timestamps, get_monthly_timestamps, choices_sanity_check, \
    load_geotiff, get_grid_idx, load_target_shp, compute_mask


def normalize(array):
    array_min, array_max = array.min(), 2000
    return (array - array_min) / (array_max - array_min + 1e-6)


def get_rgb(raster, is_norm=True):
    # Convert to numpy arrays
    blue = raster.read(1)
    green = raster.read(2)
    red = raster.read(3)

    # Normalize bands into 0.0 - 1.0 scale
    if is_norm:
        blue = normalize(blue)
        green = normalize(green)
        red = normalize(red)

    # Stack bands
    img_rgb = np.dstack((red, green, blue))

    return img_rgb


def show_true_color(raster, is_norm=True, save_path=None):
    # get rgb image
    img_rgb = get_rgb(raster, is_norm=is_norm)

    # View the color composite
    _ = plt.subplots(figsize=(7, 7))
    plt.imshow(img_rgb)
    if save_path is not None:
        plt.savefig(save_path)


def read_n_show_true_color(raster, is_norm=True, save_path=None):
    # get rgb image
    img_rgb = get_rgb(raster, is_norm=is_norm)

    # View the color composite
    _ = plt.subplots(figsize=(7, 7))
    plt.imshow(img_rgb)
    if save_path is not None:
        plt.savefig(save_path)


def get_mycmap(num_colors, cmap='Set1'):
    if num_colors < len(mpl.cm.get_cmap(cmap).colors):
        colors_list = [np.array([1, 1, 1])]
        [colors_list.append(np.array(c)) for c in mpl.cm.get_cmap(cmap).colors[:num_colors]]
        mycmap = mpl.colors.ListedColormap(colors_list)
        mycmap.set_under(color='white')
        mycmap.set_bad(color='black')
        return mycmap
    else:
        return 'Set too many colors'


def map_values_to_colors(values, mycmap):
    # map values
    norm = mpl.colors.Normalize(vmin=0, vmax=2, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=mycmap)
    mapped = mapper.to_rgba(values)

    # get the mapping of values to colors
    maps = {}
    for v in range(int(mapper.get_clim()[0]), int(mapper.get_clim()[1]) + 1):
        maps[v] = mapper.get_cmap().colors[v]

    return mapped, maps


def show_mask(classes_array, region_mask, title, windows=None, save_path=None):
    # get my color map
    mycmap = get_mycmap(2)

    # get the mapped colors
    mapped_colors, v2c = map_values_to_colors(classes_array, mycmap)
    mapped_colors[region_mask == 0] = np.array([0, 0, 0, 1])

    # show images
    _ = plt.subplots(figsize=(10, 8))
    if windows is None:
        im = plt.imshow(mapped_colors, cmap=mycmap, interpolation=None)
    else:
        # TODO: pass slice as arguments
        im = plt.imshow(mapped_colors, cmap=mycmap, interpolation=None)
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=v2c[v], label=f"{v}") for v in v2c]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
        print(f'Saved mask to {save_path}')


def show_sat_and_mask(img_filepath, pred, meta_src, region_mask=None, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 7))

    # plot true color images
    img = rasterio.open(img_filepath)
    img_rgb = get_rgb(img)
    axs[0].imshow(img_rgb)

    # plot predicted results
    mycmap = get_mycmap(2)
    mapped_colors, v2c = map_values_to_colors(pred.reshape(meta_src['height'], meta_src['width']), mycmap)
    if region_mask is not None:
        mapped_colors[region_mask == 0] = np.array([0, 0, 0, 1])
    axs[1].imshow(mapped_colors, cmap=mycmap, interpolation=None)

    # save side by side plot
    if save_path is not None:
        plt.savefig(save_path)
        print(f'Saved satellite image and its mask to {save_path}')


def plot_timestamps(timestamps, title=None, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 0.4))
    plt.plot_date(timestamps, np.ones(len(timestamps)), '|', markersize=20)
    plt.xlim(datetime.datetime(2020, 1, 1), datetime.datetime(2020, 12, 31))
    ax.axes.get_yaxis().set_visible(False)
    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Saved time stamps to {save_path}')


def plot_ndvi_profile(ndvi_array, train_mask, timestamps_ref, title=None, save_path=None):
    """
    colors:
    0 - black - no labels
    1 - red - apples
    2 - green - other crops
    3 - blue - non crops

    :param ndvi_array: np.array
        shape (height * width, ndvi time profile)
    :return:
    """
    _ = plt.subplots(1, 1, figsize=(10, 7))
    labels = np.unique(train_mask)
    print(f"labels = {labels}")
    colors_map = {0: 'black', 1: 'tab:red', 2: 'tab:green', 3: 'tab:brown'}
    labels_map = {0: 'no labels', 1: 'apples', 2: 'other crops', 3: 'non crops'}
    mean_df = pd.DataFrame()
    for label in labels:
        mean = ndvi_array[train_mask == label].mean(axis=0)
        std = ndvi_array[train_mask == label].std(axis=0)
        plt.plot(timestamps_ref, mean, color=colors_map[label], label=labels_map[label])
        plt.fill_between(timestamps_ref, mean - std, mean + std, color=colors_map[label], alpha=0.2)
        mean_df['mean_' + str(label)] = mean
        mean_df['std_' + str(label)] = std
    mean_df.to_csv(f'../data/{title}.csv', index=False)
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('NDVI')
    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
        print(f'Saved ndvi profile to {save_path}')


class NVDI_profile(object):
    """
    This is a class for plotting different kinds of NDVI profiles.
    When aggregating, always take the maximal value within a period.
    """

    def __init__(self, logger, from_dir, label_shp):
        self.logger = logger
        self.logger.info('--- NDVI profile ---')
        self.ndvi_array_raw, self.timestamps_raw, self.meta = \
            self.stack_valid_raw_ndvi(from_dir)
        # get labels
        _, train_rc_polygons, train_class_list = \
            load_target_shp(label_shp, transform=self.meta['transform'],
                            proj_out=pyproj.Proj(self.meta['crs']))
        train_mask = compute_mask(train_rc_polygons, self.meta, train_class_list)
        self.labels = train_mask.reshape(-1)

    def raw_profile(self):
        # TODO: debug, each color has three lines
        self.logger.info('Plotting raw NDVI profile...')
        plot_ndvi_profile(self.ndvi_array_raw, self.labels, self.timestamps_raw,
                          title='NDVI raw profile', save_path='../figs/NDVI_raw.png')

    def weekly_profile(self, interpolation='previous'):
        self.logger.info('Plotting weekly NDVI profile...')
        ndvi_array_weekly, _, timestamps_weekly_ref = \
            self.stack_equidistant_ndvi('weekly', interpolation)
        plot_ndvi_profile(ndvi_array_weekly, self.labels, timestamps_weekly_ref,
                          title='NDVI weekly profile', save_path='../figs/NDVI_weekly.png')

    def monthly_profile(self, interpolation='previous'):
        self.logger.info('Plotting monthly NDVI profile...')
        ndvi_array_monthly, _, timestamps_monthly_ref = \
            self.stack_equidistant_ndvi('monthly', interpolation)
        plot_ndvi_profile(ndvi_array_monthly, self.labels, timestamps_monthly_ref,
                          title='NDVI monthly profile', save_path='../figs/NDVI_monthly.png')

    def stack_valid_raw_ndvi(self, from_dir):
        bands_list_raw, timestamps_raw, timestamps_missing, meta = \
            self.stack_valid_raw_timestamps(self.logger, from_dir)
        bands_array_raw = add_bands(self.logger, np.stack(bands_list_raw, axis=2), new_bands_name=['ndvi'])
        ndvi_array_raw = bands_array_raw[:, -1, :]
        return ndvi_array_raw, timestamps_raw, meta

    @staticmethod
    def stack_valid_raw_timestamps(logger, from_dir):
        logger.info('Stacking valid raw timestamps...')
        bands_list_valid, timestamps_valid, timestamps_missing = [], [], []
        for filename in sorted([f for f in os.listdir(from_dir) if f.endswith('tiff')]):
            raster_filepath = from_dir + filename
            band, meta = load_geotiff(raster_filepath)
            # pixel values check
            if np.array(band).mean() != 0.0:
                bands_list_valid.append(np.stack(band, axis=2).reshape(-1, len(band)))
                timestamps_valid.append(datetime.datetime.strptime(re.split('[_.]', filename)[-2],
                                                                   '%Y%m%dT%H%M%S%f').date())
            else:
                timestamps_missing.append(datetime.datetime.strptime(re.split('[_.]', filename)[-2],
                                                                     '%Y%m%dT%H%M%S%f').date())
                logger.info(f'  Discard {filename} due to empty value.')
        logger.info('Stack done!')

        return bands_list_valid, timestamps_valid, timestamps_missing, meta

    def stack_equidistant_ndvi(self, way, interpolation='previous'):
        choices_sanity_check(['weekly', 'monthly'], way, 'way')
        if way == 'weekly':
            timestamps_af = [ts - datetime.timedelta(days=ts.weekday()) for ts in
                             self.timestamps_raw]  # datetime.weekday() returns 0~6
            timestamps_ref = get_weekly_timestamps()
        else:
            timestamps_af = [ts - datetime.timedelta(days=ts.day - 1) for ts in
                             self.timestamps_raw]  # datetime.day returns 1-31
            timestamps_ref = get_monthly_timestamps()

        # stack equidistantly
        timestamps_af_pd = pd.Series(timestamps_af)
        ndvi_list_eql = []
        for i, timestamp in enumerate(timestamps_ref, start=1):
            # get all the indices
            ids = list(timestamps_af_pd[timestamps_af_pd.eq(timestamp)].index)
            # with non-empty data
            if len(ids) != 0:
                ndvi_array = self.ndvi_array_raw[:, ids].max(axis=1)
                # format printing
                print_str = ''
                for id in ids:
                    print_str += f'{self.timestamps_raw[id].strftime("%Y-%m-%d")}, '
                self.logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} ({print_str})')
            else:
                if interpolation == 'zero' or i == 1:
                    ndvi_array = np.zeros(self.meta['height'] * self.meta['width'])
                    self.logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} (0)')
                else:
                    ndvi_array = ndvi_list_eql[-1]
                    self.logger.info(f'  [{i}/{len(timestamps_ref)}] {timestamp} (previous)')
            ndvi_list_eql.append(ndvi_array)
        ndvi_array_eql = np.stack(ndvi_list_eql, axis=1)

        return ndvi_array_eql, timestamps_af, timestamps_ref


def visualize_train_test_grid_split(meta_src, spatial_dict, grid_idx_test, grid_idx_fold, save_path):
    # get the grid idx
    grid_size, height, width = spatial_dict['grid_size'], spatial_dict['height'], spatial_dict['width']
    grid_idx = get_grid_idx(grid_size, height, width).reshape(-1)
    output = np.zeros_like(grid_idx)
    # build test
    unique_grid_idx_test = list(set(grid_idx_test))
    test_mask = [True if idx in unique_grid_idx_test else False for idx in grid_idx]
    output[test_mask] = 5
    # build train and val
    if len(grid_idx_fold) == 1:
        unique_grid_idx_train_val = list(set(grid_idx_fold[0]))
        train_val_mask = [True if idx in unique_grid_idx_train_val else False for idx in grid_idx]
        output[train_val_mask] = 1
    else:
        for i, fold in enumerate(grid_idx_fold, start=1):
            unique_grid_idx_val = list(set(fold))
            val_mask = [True if idx in unique_grid_idx_val else False for idx in grid_idx]
            output[val_mask] = i
    color_map = {
        0: (0, 0, 0),
        5: (138, 205, 226),  # test
        1: (239, 135, 190),  # train (main color)
        2: (249, 163, 203),
        3: (252, 188, 215),
        4: (255, 206, 230),
    }

    with rasterio.Env():
        # Write an array as a raster band to a new 8-bit file. We start with the profile of the source
        out_meta = meta_src.copy()
        out_meta.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw')
        with rasterio.open(save_path, 'w', **out_meta) as dst:
            # reshape into (band, height, width)
            dst.write(output.reshape(1, out_meta['height'], out_meta['width']).astype(rasterio.uint8))
            dst.write_colormap(1, color_map)
