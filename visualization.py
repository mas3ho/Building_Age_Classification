import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

from utils import BASE_BANDS, CLASS_COLORS, CLASS_LABELS
from feature_engineering import get_pixel_ts, get_yearly_means


def plot_feature_class_distribution(features, labels, colors):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, data) in zip(axes, features.items()):
        classes = [1, 2, 3, 4]
        counts = [(data["age_class"] == cls).sum() for cls in classes]
        names = [f"Class {cls}\n{labels[name][cls]}" for cls in classes]
        bars = ax.bar(names, counts, color=[colors[cls] for cls in classes])
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_ylabel("Pixel count")
        ax.set_title(f"{name}: class distribution", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_cross_validation_results(scores, true_labels, predicted_labels, labels):
    from sklearn.metrics import confusion_matrix

    folds = len(scores)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    x_values = np.arange(1, folds + 1)
    axes[0].scatter(x_values, scores, color="steelblue", s=40, alpha=0.8)
    axes[0].axhline(
        mean_score,
        color="navy",
        linestyle="--",
        label=f"Mean = {mean_score:.4f}",
    )
    axes[0].fill_between(
        [0.5, folds + 0.5],
        mean_score - std_score,
        mean_score + std_score,
        color="steelblue",
        alpha=0.15,
        label=f"Std = {std_score:.4f}",
    )
    axes[0].set_xlabel("Fit number")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Madrid repeated cross-validation", fontweight="bold")
    axes[0].legend(fontsize=9)

    classes = sorted(np.unique(true_labels))
    matrix = confusion_matrix(true_labels, predicted_labels, labels=classes)
    normalized = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)
    image = axes[1].imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    tick_labels = [f"C{cls}\n{labels[cls]}" for cls in classes]
    axes[1].set_xticks(range(len(classes)), tick_labels, fontsize=8)
    axes[1].set_yticks(range(len(classes)), tick_labels, fontsize=8)
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = normalized[row, column]
            axes[1].text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.6 else "black",
            )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("OOF confusion matrix", fontweight="bold")
    plt.colorbar(image, ax=axes[1])
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importances, feature_names, top_n=20):
    order = np.argsort(importances)[::-1][:top_n]
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(
        range(len(order)),
        importances[order][::-1],
        color="steelblue",
        edgecolor="none",
    )
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels([feature_names[index] for index in order][::-1], fontsize=9)
    axis.set_xlabel("Mean decrease in impurity")
    axis.set_title(f"Top {top_n} features - Random Forest", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_labels, title):
    from sklearn.metrics import confusion_matrix

    classes = sorted(np.unique(y_true))
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    normalized = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    tick_labels = [f"C{cls}\n{class_labels[cls]}" for cls in classes]
    axis.set_xticks(range(len(classes)), tick_labels, fontsize=8)
    axis.set_yticks(range(len(classes)), tick_labels, fontsize=8)
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.6 else "black",
            )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title, fontweight="bold")
    plt.colorbar(image, ax=axis)
    plt.tight_layout()
    plt.show()


def plot_prototype_results(results, zero_shot_f1, madrid_f1):
    shots = list(results)
    means = [np.mean(results[value]) for value in shots]
    standard_deviations = [np.std(results[value]) for value in shots]
    figure, axis = plt.subplots(figsize=(10, 5))
    for value in shots:
        axis.scatter(
            [value] * len(results[value]),
            results[value],
            color="#fd8d3c",
            alpha=0.35,
            s=20,
        )
    axis.errorbar(
        shots,
        means,
        yerr=standard_deviations,
        fmt="o-",
        color="#fd8d3c",
        capsize=4,
        label="Prototype transfer",
    )
    axis.axhline(zero_shot_f1, color="steelblue", linestyle="--", label="Zero-shot")
    axis.axhline(madrid_f1, color="green", linestyle="--", label="Madrid CV")
    axis.set_xlabel("Labeled Amsterdam examples per class")
    axis.set_ylabel("Macro F1")
    axis.set_xscale("log")
    axis.set_title("Few-shot prototype transfer", fontweight="bold")
    axis.legend()
    plt.tight_layout()
    plt.show()


def make_pixel_grid(pixel_ids, values, fill_value):
    px = pixel_ids[:, 0].astype(int)
    py = pixel_ids[:, 1].astype(int)
    px -= px.min()
    py -= py.min()
    height, width = py.max() + 1, px.max() + 1
    shape = (height, width) if values.ndim == 1 else (height, width, values.shape[1])
    image = np.full(shape, fill_value, dtype=np.float32)
    image[py, px] = values
    return image


def plot_spatial_uncertainty(pixel_ids, probabilities, true_labels, title):
    eps = 1e-10
    rgb = (np.floor(probabilities[:, 1:] * 256).clip(0, 255) / 255.0).astype(float)
    class_colors = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=float,
    )
    class_image = make_pixel_grid(
        pixel_ids,
        class_colors[true_labels - 1],
        fill_value=0.7,
    )
    probability_image = make_pixel_grid(pixel_ids, rgb, fill_value=0.7)
    entropy = -(probabilities * np.log2(probabilities + eps)).sum(axis=1)
    true_probability = probabilities[np.arange(len(true_labels)), true_labels - 1]
    kl = -np.log2(true_probability + eps)
    entropy_image = make_pixel_grid(pixel_ids, entropy, fill_value=np.nan)
    kl_image = make_pixel_grid(pixel_ids, kl, fill_value=np.nan)

    figure, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes[0, 0].imshow(probability_image)
    axes[0, 0].set_title(f"{title}\nPredicted probabilities")
    axes[0, 1].imshow(class_image)
    axes[0, 1].set_title(f"{title}\nActual classes")
    entropy_plot = axes[1, 0].imshow(entropy_image, cmap="hot_r", vmin=0, vmax=2)
    kl_plot = axes[1, 1].imshow(kl_image, cmap="hot_r", vmin=0, vmax=2)
    axes[1, 0].set_title("Shannon entropy")
    axes[1, 1].set_title("KL from true class")
    for axis in axes.flat:
        axis.axis("off")
    figure.colorbar(entropy_plot, ax=axes[1, 0], fraction=0.046, pad=0.04)
    figure.colorbar(kl_plot, ax=axes[1, 1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_spatial_accuracy(pixel_ids, true_labels, predicted_labels, title):
    correct = true_labels == predicted_labels
    correct_image = make_pixel_grid(
        pixel_ids,
        np.where(correct[:, None], [[0, 0, 0]], [[1, 0, 0]]),
        fill_value=0.6,
    )
    colors = np.array(
        [[0, 0, 0], [140, 0, 0], [0, 140, 0], [0, 0, 140]],
        dtype=float,
    ) / 255.0
    class_image = make_pixel_grid(
        pixel_ids,
        np.where(correct[:, None], colors[true_labels - 1], [[1, 1, 0]]),
        fill_value=0.6,
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(correct_image)
    axes[0].set_title(f"{title}\nCorrect vs incorrect")
    axes[1].imshow(class_image)
    axes[1].set_title(f"{title}\nCorrect class / incorrect")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def plot_coverage_distribution(data, title, threshold=0.15, bins=20):
    plt.figure(figsize=(7, 4))
    plt.hist(data["coverage"], bins=bins, color="#4393c3", alpha=0.8)
    plt.axvline(threshold, color="red", linestyle="--", label=f"threshold = {threshold:.2f}")
    plt.xlabel("Building coverage")
    plt.ylabel("Rows")
    plt.title(f"{title} - building coverage")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_class_distribution(data, ax=None, title=""):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    counts = data.drop_duplicates(["px_key", "py_key"])["age_class"].value_counts().sort_index()
    labels = CLASS_LABELS.get(title, CLASS_LABELS["Madrid"])
    ax.bar(
        [f"Class {int(cls)}\n{labels.get(int(cls), '')}" for cls in counts.index],
        counts.values,
        color=[CLASS_COLORS[int(cls)] for cls in counts.index],
    )
    ax.set_title(f"{title} - age class distribution")
    ax.set_ylabel("Unique pixel locations")
    return ax


def plot_dodged_hist(data, value_column, class_column="age_class", ax=None, bins=30):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for cls in sorted(data[class_column].dropna().unique()):
        values = data.loc[data[class_column] == cls, value_column].dropna()
        ax.hist(
            values,
            bins=bins,
            alpha=0.45,
            color=CLASS_COLORS[int(cls)],
            label=f"Class {int(cls)}",
        )
    ax.set_xlabel(value_column)
    ax.set_ylabel("Pixels")
    ax.legend()
    return ax


def plot_pixel_ts(band, flat, class_pixels, year_min, year_max):
    fig, ax = plt.subplots(figsize=(10, 5))
    for cls, pixels in class_pixels.items():
        for pixel in pixels:
            series = get_pixel_ts(flat, pixel, band)
            ax.plot(series["year"], series[band], color=CLASS_COLORS[cls], alpha=0.65)
    ax.set_xlim(year_min, year_max)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{band} reflectance")
    ax.set_title(f"{band} pixel trajectories")
    plt.tight_layout()
    plt.show()


def plot_yearly_class_means(band, flat, year_min, year_max):
    fig, ax = plt.subplots(figsize=(10, 5))
    yearly = get_yearly_means(flat, band)
    for cls in sorted(yearly["age_class"].dropna().unique()):
        subset = yearly[yearly["age_class"] == cls]
        ax.plot(subset["year"], subset["mean"], color=CLASS_COLORS[int(cls)])
        ax.fill_between(
            subset["year"],
            subset["mean"] - subset["std"],
            subset["mean"] + subset["std"],
            color=CLASS_COLORS[int(cls)],
            alpha=0.15,
        )
    ax.set_xlim(year_min, year_max)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{band} reflectance")
    ax.set_title(f"{band} yearly class means")
    plt.tight_layout()
    plt.show()


def plot_metric_grid(cities, pixel_stats, metric_template, title, ylabel):
    fig, axes = plt.subplots(1, len(BASE_BANDS), figsize=(18, 4), squeeze=False)
    for ax, band in zip(axes[0], BASE_BANDS):
        column = metric_template.format(band=band)
        for cls in sorted(pixel_stats[cities[0][0]]["age_class"].dropna().unique()):
            values = []
            for _, stats, _ in cities:
                values.extend(stats.loc[stats["age_class"] == cls, column].dropna())
            if values:
                ax.hist(values, bins=30, alpha=0.45, color=CLASS_COLORS[int(cls)])
        ax.set_title(band)
        ax.set_xlabel(column)
    axes[0, 0].set_ylabel(ylabel)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_scatter_grid(cities, pixel_stats, x_template, y_template, title, xlabel, ylabel):
    fig, axes = plt.subplots(1, len(BASE_BANDS), figsize=(18, 4), squeeze=False)
    for ax, band in zip(axes[0], BASE_BANDS):
        x_col = x_template.format(band=band)
        y_col = y_template.format(band=band)
        for city, stats, _ in cities:
            ax.scatter(stats[x_col], stats[y_col], s=4, alpha=0.2, label=city)
        ax.set_title(band)
        ax.set_xlabel(xlabel)
    axes[0, 0].set_ylabel(ylabel)
    axes[0, -1].legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
