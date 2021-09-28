import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import rasterio
from datetime import datetime


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
    fig = plt.subplots(figsize=(7, 7))
    plt.imshow(img_rgb)
    if save_path is not None:
        plt.savefig(save_path)


def read_n_show_true_color(raster, is_norm=True, save_path=None):
    # get rgb image
    img_rgb = get_rgb(raster, is_norm=is_norm)

    # View the color composite
    fig = plt.subplots(figsize=(7, 7))
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
    fig = plt.subplots(figsize=(10, 8))
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


# def plot_feature_importance(columns, feature_importance, save_path=None):
#     plt.bar(columns, feature_importance)
#     plt.xticks(rotation = 45)
#     if save_path is not None:
#         plt.savefig(save_path)


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


def plot_timestamps(timestamps, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 0.4))
    plt.plot_date(timestamps, np.ones(len(timestamps)), '|', markersize=20)
    plt.xlim(datetime(2020, 1, 1), datetime(2020, 12, 31))
    ax.axes.get_yaxis().set_visible(False)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Saved time stamps to {save_path}')


# def plot_ndvi_profile(df_ndvi, timestamps_weekly, save_path=None):
#     """
#
#     :param df_ndvi: DataFrame
#         shape (height * width, ndvi time profile)
#     :return:
#     """
#     _ = plt.subplots(1, 1, figsize=(10, 7))
#     labels = np.unique(df_ndvi['label'].values)
#     colors_map = {0: 'black', 1: 'red', 2: 'green', 3: 'blue'}
#     for label in labels:
#         ndvi_label = df_ndvi[df_ndvi['label'] == label].iloc[:, :-1]
#         for i in random.sample(range(ndvi_label.shape[0]), 100):
#             plt.plot(timestamps_weekly, ndvi_label.iloc[i, :], color=colors_map[label], lw=0.5, alpha=0.2)
#         plt.plot(timestamps_weekly, ndvi_label.mean(axis=0), color=colors_map[label], lw=2)
#     if save_path is not None:
#         plt.savefig(save_path)
#         print(f'Saved ndvi profile to {save_path}')


def plot_ndvi_profile(ndvi_array, train_mask, timestamps_weekly, save_path=None):
    """

    :param ndvi_array: np.array
        shape (height * width, ndvi time profile)
    :return:
    """
    _ = plt.subplots(1, 1, figsize=(10, 7))
    labels = np.unique(train_mask)
    colors_map = {0: 'black', 1: 'red', 2: 'green', 3: 'blue'}
    for label in labels:
        ndvi_label = ndvi_array[train_mask == label]
        for i in random.sample(range(ndvi_label.shape[0]), 100):
            plt.plot(timestamps_weekly, ndvi_label[i, :], color=colors_map[label], lw=0.5, alpha=0.2)
        plt.plot(timestamps_weekly, ndvi_label.mean(axis=0), color=colors_map[label], lw=2)
    if save_path is not None:
        plt.savefig(save_path)
        print(f'Saved ndvi profile to {save_path}')
