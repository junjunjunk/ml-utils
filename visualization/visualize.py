# Basic Package
from typing import Any, List, Tuple

# Visualize Package
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn2

# Setting for Packages
plt.style.use("seaborn-whitegrid")
sns.set_style("white")
color_map = plt.get_cmap("tab10")


def visualize_venn(
    train_df: pd.DataFrame, test_df: pd.DataFrame, plot_column: str
) -> Tuple[plt.Figure, plt.Axes]:
    """Make venn figure.

    Args:
        train_df (pd.DataFrame): Dataframe of train data
        test_df (pd.DataFrame): Dataframe of test data
        plot_column (str): A specified categorical column for visualization.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Figure & Axes
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    venn2(
        subsets=(
            set(train_df[plot_column].unique()),
            set(test_df[plot_column].unique()),
        ),
        set_labels=("Train", "Test"),
        color=color_map,
        ax=ax,
    )
    ax.set_title(f"{plot_column}")
    fig.tight_layout()
    return fig, ax


def visualize_compare_distribution(
    x: List[float], y: List[float], x_label: str, y_label: str
) -> Tuple[plt.Figure, plt.Axes]:
    """Make distribution figure of x,y for comparison.

    Args:
        x (List[float]): A list of number which wanted to plot the distribution
        y (List[float]): A list of number which wanted to plot the distribution
        x_label (str): x's label text
        y_label (str): y's label text

    Returns:
        Tuple[plt.Figure, plt.Axes]: Figure & Axes
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(x, label=x_label, ax=ax, color="black")
    sns.histplot(y, label=y_label, ax=ax, color="C1")
    ax.set_title("Two distribution")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    return fig, ax


def visualize_importance(
    models: List[Any], train_columns: pd.DataFrame, top: int = 50, model_type="lgbm"
) -> Tuple[plt.Figure, plt.Axes]:
    """Make figure of feature importances of models.

    CV's diffs are plotted with boxenplot.
    Plot ascending by feature_importance.

    Args:
        models (List[Any]): A list of models
        train_columns (pd.DataFrame): A list of columns used to training model
        top (int, optional): A cutoff ranking number for plot importances. Defaults to 50.
        model_type (str, optional): Model type name. Defaults to 'lgbm'.

    Raises:
        ValueError: When unsupported models are given.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Figure & Axes
    """
    if model_type == "lgbm":
        feature_importance_df = pd.DataFrame()
        for i, model in enumerate(models):
            _df = pd.DataFrame()
            _df["feature_importance"] = model.feature_importance()
            _df["column"] = train_columns  # feat_train_df.columns
            _df["fold"] = i + 1
            feature_importance_df = pd.concat(
                [feature_importance_df, _df], axis=0, ignore_index=True
            )

        order = (
            feature_importance_df.groupby("column")
            .sum()[["feature_importance"]]
            .sort_values("feature_importance", ascending=False)
            .index[:top]
        )

        fig, ax = plt.subplots(figsize=(8, max(6, len(order) * 0.25)))
        sns.boxenplot(
            data=feature_importance_df,
            x="feature_importance",
            y="column",
            order=order,
            ax=ax,
            palette="viridis",
            orient="h",
        )
        ax.tick_params(axis="x", rotation=90)
        ax.set_title("Importance")
        ax.grid()
        fig.tight_layout()
        return fig, ax
    raise ValueError("These models is not supported.")


def visualize_mulitiple_time_series(
    df: pd.DataFrame, x: str, y1: str, y2: str = None
) -> Tuple[plt.Figure, plt.Axes]:
    """make figure of time series.(single or multiple)

    Args:
        df (pd.DataFrame): [description]
        x (str): x-axis column name
        y1 (str): y-axis column name
        y2 (str): y-axis column name

    Returns:
        Tuple[plt.Figure, plt.Axes]: Figure & Axes
    """

    fig, ax1 = plt.subplots(1, 1, figsize=(16, 9), dpi=80)

    # Plot Line1 (Left Y Axis)
    ax1.plot(df[x], df[y1], color="tab:red")
    ax1.set_xlabel(x, fontsize=20)
    ax1.tick_params(axis="x", rotation=0, labelsize=12)
    ax1.set_ylabel(y1, color="tab:red", fontsize=20)
    ax1.tick_params(axis="y", rotation=0, labelcolor="tab:red")
    ax1.grid(alpha=0.4)

    # Plot Line2 (Right Y Axis)
    if y2 is not None:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(df[x], df[y2], color="tab:blue")
        ax2.set_ylabel(y2, color="tab:blue", fontsize=20)
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax2.set_xticks(np.arange(0, len(x), 60))
        ax2.set_xticklabels(x[::60], rotation=90, fontdict={"fontsize": 10})

    fig.tight_layout()
    return fig, ax1
