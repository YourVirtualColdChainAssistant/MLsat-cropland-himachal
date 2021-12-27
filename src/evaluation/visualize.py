import os
import re
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import rasterio
import shapely
import pyproj
import pandas as pd
import datetime


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
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def read_n_show_true_color(raster, is_norm=True, save_path=None):
    # get rgb image
    img_rgb = get_rgb(raster, is_norm=is_norm)

    # View the color composite
    _ = plt.subplots(figsize=(7, 7))
    plt.imshow(img_rgb)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


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
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Saved mask to {save_path}')
    plt.close()


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
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Saved satellite image and its mask to {save_path}')
    plt.close()


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
    plt.close()


def plot_profile(data, label, timestamps, veg_index, title=None, save_path=None):
    """
    Plot NDVI / GNDVI / ... profile of classes as time evolving.

    Parameters
    ----------
    data: np.array
        shape (n_data, n_weeks)
    label: np.array
        shape (n_data, )
    timestamps: list
        A list of dates (of each Monday in 2020).
    veg_index: string
        The name of the vegetation index to be drew.
    title: string
        Name of plot.
    save_path: string
        Path to store this plot.

    Returns
    -------

    """
    _ = plt.subplots(1, 1, figsize=(10, 7))
    unique_label = np.unique(label)
    veg_index = veg_index.upper()
    colors_map = {0: 'black', 1: 'tab:red', 2: 'tab:green', 3: 'tab:brown'}
    labels_map = {0: 'unlabeled', 1: 'apples', 2: 'other croplands', 3: 'non-croplands'}

    df = pd.DataFrame()
    for l in unique_label:
        mean = data[label == l].mean(axis=0)
        std = data[label == l].std(axis=0)
        plt.plot(timestamps, mean, color=colors_map[l], label=labels_map[l])
        plt.fill_between(timestamps, mean - std, mean + std, color=colors_map[l], alpha=0.2)
        df['mean_' + str(l)] = mean
        df['std_' + str(l)] = std
    plt.legend(loc='best')
    plt.ylim(0, 1)
    plt.xlabel('Time')
    plt.ylabel(veg_index)
    if title is not None:
        plt.title(title)
        df.to_csv(f'./figs/{title}.csv', index=False)
    else:
        df.to_csv(f'./figs/{veg_index}_profile.csv', index=False)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Saved {veg_index} profile to {save_path}')
    plt.close()


def visualize_cv_fold(grid, meta, save_path):
    shapes = iter([(shapely.geometry.mapping(poly), v + 1) for poly, v in zip(grid.geometry, grid.fold_id)])
    fold_img = rasterio.features.rasterize(shapes, out_shape=(meta['height'], meta['width']),
                                           transform=meta['transform'])
    out_meta = meta.copy()
    out_meta.update(count=1, dtype=rasterio.uint8)
    write_1_band_raster(fold_img, out_meta, save_path)


def visualize_cv_polygons(scv, train_val_coords, meta, save_path):
    excluded_indices = []
    all_indices = np.arange(train_val_coords.shape[0])
    # construct the image
    img = np.zeros((meta['height'], meta['width']), dtype=int)
    for i, (train_indice, val_indice) in enumerate(scv.split(train_val_coords), start=1):
        # each fold is a color
        xs = train_val_coords.loc[val_indice, 'geometry'].apply(lambda x: x.coords[:][0][0]).to_numpy()
        ys = train_val_coords.loc[val_indice, 'geometry'].apply(lambda x: x.coords[:][0][1]).to_numpy()
        rows, cols = rasterio.transform.rowcol(meta['transform'], xs, ys)
        img[rows, cols] = i
        print(f'  fold {i} = {i}')
        excluded_indices.append(list(set(all_indices) - set(train_indice) - set(val_indice)))
    excluded_indices = list(itertools.chain.from_iterable(excluded_indices))
    # excluded points is another color
    if len(excluded_indices) != 0:
        xs = train_val_coords.loc[excluded_indices, 'geometry'].apply(lambda x: x.coords[:][0][0]).to_numpy()
        ys = train_val_coords.loc[excluded_indices, 'geometry'].apply(lambda x: x.coords[:][0][1]).to_numpy()
        rows, cols = rasterio.transform.rowcol(meta['transform'], xs, ys)
        img[rows, cols] = i + 1
        print(f'  Excluded points = {i + 1}')
    # save new raster
    out_meta = meta.copy()
    out_meta.update(dtype=rasterio.uint8, count=1)
    write_1_band_raster(img, out_meta, save_path)


def write_1_band_raster(img, out_meta, save_path, cmap='Set2'):
    color_map = {v: tuple(255 * np.array(c)) for v, c in enumerate(mpl.cm.get_cmap(cmap).colors, start=1)}
    with rasterio.open(save_path, 'w', **out_meta) as dst:
        # reshape into (band, height, width)
        dst.write(img.astype(rasterio.uint8), indexes=1)
        dst.write_colormap(1, color_map)
