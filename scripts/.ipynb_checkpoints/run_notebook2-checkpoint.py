from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feature_engineering import (
    compute_pixel_stats,
    find_best_pixels,
    make_flat_view,
)
from load_data import load_city_data
from utils import BASE_BANDS, CLASS_COLORS, CLASS_LABELS, default_data_paths
from visualization import (
    plot_class_distribution,
    plot_coverage_distribution,
    plot_metric_grid,
    plot_pixel_ts,
    plot_scatter_grid,
    plot_yearly_class_means,
)


def main():
    plt.rcParams["figure.dpi"] = 100
    madrid_path, amsterdam_path = default_data_paths(Path("data"))
    madrid = load_city_data(madrid_path)
    amsterdam = load_city_data(amsterdam_path)
    madrid_flat = make_flat_view(madrid)
    amsterdam_flat = make_flat_view(amsterdam)
    cities = [
        ("Madrid", madrid_flat, madrid),
        ("Amsterdam", amsterdam_flat, amsterdam),
    ]
    year_min = min(flat["year"].min() for _, flat, _ in cities)
    year_max = max(flat["year"].max() for _, flat, _ in cities)

    for name, data in [("Madrid", madrid), ("Amsterdam", amsterdam)]:
        print(f"\n{name}: {len(data):,} rows x {data.shape[1]} columns")
        print(f"Years: {data['year'].min()} - {data['year'].max()}")
        print(f"Locations: {data.groupby(['px_key', 'py_key']).ngroups:,}")
        pixels = data.drop_duplicates(["px_key", "py_key"])
        for cls, count in pixels["age_class"].value_counts().sort_index().items():
            print(f"  Class {int(cls)} ({CLASS_LABELS[name][int(cls)]}): {count:,}")

    print("\nCoverage")
    for name, _, raw in cities:
        print(
            f"{name}: range={raw['coverage'].min():.2f}-{raw['coverage'].max():.2f}, "
            f"mean={raw['coverage'].mean():.2f}, "
            f"median={raw['coverage'].median():.2f}"
        )
        plot_coverage_distribution(raw, title=name, threshold=0.15, bins=20)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    for ax, (name, flat, _) in zip(axes, cities):
        plot_class_distribution(flat, ax=ax, title=name)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=False)
    bins = np.arange(1840, 2031, 5)
    for ax, (name, flat, _) in zip(axes, cities):
        locations = flat.drop_duplicates(["px_key", "py_key"])
        for cls in [1, 2, 3, 4]:
            subset = locations[locations["age_class"] == cls]["weighted_mean_year"]
            if subset.empty:
                continue
            ax.hist(
                subset,
                bins=bins,
                alpha=0.55,
                color=CLASS_COLORS[cls],
                label=f"Class {cls}: {CLASS_LABELS[name][cls]} (n={len(subset):,})",
            )
        ax.set_xlabel("weighted_mean_year")
        ax.set_ylabel("Unique pixel locations")
        ax.set_title(f"{name} construction year distribution", fontweight="bold")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    class_pixels = {}
    for name, flat, _ in cities:
        class_pixels[name] = {}
        for cls in [1, 2, 3, 4]:
            pixels = find_best_pixels(flat, cls, n=3)
            class_pixels[name][cls] = pixels
            print(f"{name} class {cls} sample pixels: {pixels}")
        for band in ["Blue", "NIR", "SWIR1"]:
            plot_pixel_ts(band, flat, class_pixels[name], year_min, year_max)
        for band in ["Blue", "NIR", "SWIR1"]:
            plot_yearly_class_means(band, flat, year_min, year_max)

    pixel_stats = {}
    for name, flat, _ in cities:
        pixel_stats[name] = compute_pixel_stats(flat)
        print(f"{name}: {len(pixel_stats[name]):,} pixels")

    stats_cities = [
        (name, pixel_stats[name], raw)
        for name, _, raw in cities
    ]
    for template, title, ylabel in [
        ("{band}_mean", "Per-pixel mean reflectance by age class", "Mean reflectance"),
        ("{band}_std", "Per-pixel std dev of reflectance by age class", "Std dev"),
        ("d_{band}_mean", "Per-pixel mean year-on-year change by age class", "Mean annual change"),
        ("d_{band}_std", "Per-pixel std dev of year-on-year change by age class", "Std dev annual change"),
    ]:
        plot_metric_grid(stats_cities, pixel_stats, template, title, ylabel)

    for x_template, y_template, title, xlabel, ylabel in [
        ("{band}_mean", "{band}_std", "Per-pixel mean vs std dev of reflectance", "Mean", "Std dev"),
        ("d_{band}_mean", "d_{band}_std", "Per-pixel mean vs std dev of change", "Mean annual change", "Std dev"),
        ("{band}_mean", "d_{band}_mean", "Reflectance mean vs annual change", "Mean reflectance", "Mean annual change"),
        ("{band}_std", "d_{band}_std", "Reflectance std dev vs change std dev", "Std dev reflectance", "Std dev annual change"),
    ]:
        plot_scatter_grid(
            stats_cities,
            pixel_stats,
            x_template,
            y_template,
            title,
            xlabel,
            ylabel,
        )

    print("\nBand summary:")
    print(madrid_flat[BASE_BANDS].describe().round(1))
    for name, flat, _ in cities:
        zero = (flat[BASE_BANDS] == 0).any(axis=1)
        saturated = (flat[BASE_BANDS] == 65535).any(axis=1)
        print(f"{name}: zero rows={zero.sum():,}, saturated rows={saturated.sum():,}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (name, flat, _) in zip(axes, cities):
        correlation = flat[BASE_BANDS].corr().values
        image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(BASE_BANDS)))
        ax.set_xticklabels(BASE_BANDS, rotation=45, ha="right")
        ax.set_yticks(range(len(BASE_BANDS)))
        ax.set_yticklabels(BASE_BANDS)
        for i in range(len(BASE_BANDS)):
            for j in range(len(BASE_BANDS)):
                ax.text(j, i, f"{correlation[i, j]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(image, ax=ax)
        ax.set_title(f"Band correlation - {name}", fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
