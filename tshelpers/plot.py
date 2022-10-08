import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Seaborn style settings
sns.set_theme(style="ticks", palette="mako")

# Auxiliar function to plot a subset
def plot_sequence(
    data,
    time="DATE_PST",
    value="RAW_VALUE",
    start=None,
    end=None,
    y_label="PM 2.5",
    x_label="Date",
    x_rotate=30,
    plot_title=None,
    plot_sup_title=None,
    font_size=12,
    figsize=(15, 5),
):
    """
    Plot times series subset within passed start and end date interval.
    data: Pandas DataFrame with a time series and a value column.
    time: DateTime variable in the dataset.
    value: Value variable or list of variables to subset with date column.
    start: Start date for interval.
    end: End date for interval.
    y_label: Plot label for Y axis.
    x_label: Plot label for X axis.
    x_rotate: Rotation angle for X axis ticks.
    plot_title: Plot title.
    plot_sup_title: Plot sub-title.
    font_size: Plot tile font size.
    figsize: Plot canvas size.
    """
    sns.set_theme(style="ticks", palette="mako")

    if start is None:
        start = min(data[time])
    if end is None:
        end = max(data[time])

    # Standard visualization with seaborn lineplot
    plt.figure(figsize=figsize)

    # Interval to plot
    if start < min(data[time]):
        print(
            "WARNING: Plot start exceeds subset limit. Truncating to subset start date..."
        )
        start = min(data[time])
    if end > max(data[time]):
        print(
            "WARNING: Plot end exceeds subset limit. Truncating to subset end date..."
        )
        end = max(data[time])
    subset = data[(data[time] >= start) & (data[time] <= end)]

    # Title and subtitle
    if plot_title is None:
        plot_title = f"Time Series sequence for {y_label}"
    if plot_sup_title is None:
        plot_sup_title = (
            f"From {start:%H %p}, {start:%d-%b-%Y} to {end:%H %p}, {end:%d-%b-%Y}"
        )

    # Lineplot
    l = sns.lineplot(x=time, y=value, data=subset)
    # Filling values to indicate NaN
    l.fill_between(x=time, y1=value, data=subset, alpha=0.65)

    l.set_title(plot_title, y=1.05, fontsize=font_size, fontweight="bold")
    l.set_ylabel(y_label)
    l.set_xlabel(x_label)
    plt.suptitle(plot_sup_title, y=0.92, fontsize=12)
    plt.xticks(rotation=x_rotate)
    plt.show()


# Auxiliar function to plot overlaying subsets
def plot_compare(
    df,
    df_missing,
    time="DATE_PST",
    value="RAW_VALUE",
    start=None,
    end=None,
    y_label="PM 2.5",
    x_label="Date",
    x_rotate=30,
    fill=True,
    missing_only=True,
    plot_title=None,
    plot_sup_title=None,
    font_size=12,
    figsize=(15, 5),
):
    """
    Plot times series subset within passed start and end date interval.
    df: Pandas DataFrame with a time series and a value column.
    df_missing: Pandas DataFrame with a time series and a value column.
    time: DateTime variable in the dataset.
    value: Value variable or list of variables to subset with date column.
    start: Start date for interval.
    end: End date for interval.
    y_label: Plot label for Y axis.
    x_label: Plot label for X axis.
    x_rotate: Rotation angle for X axis ticks.
    fill: Boolean argument to fill area underneath df1 lines.
    missing_only: Boolean arugment to plot only non-missing equivalent of df_missing.
    plot_title: Plot title.
    plot_sup_title: Plot sub-title.
    font_size: Plot tile font size.
    figsize: Plot canvas size.
    """
    sns.set_theme(style="ticks", palette="mako")
    warnings.filterwarnings("ignore", category=UserWarning)

    assert len(df) == len(df_missing), "Sequences must be of the same length."
    assert min(df[time]) == min(
        df_missing[time]
    ), "Sequences do not start at the same time stamp."
    assert max(df[time]) == max(
        df_missing[time]
    ), "Sequences do not end at the same time stamp."

    if start is None:
        start = min(df[time])
    if end is None:
        end = max(df[time])

    # Standard visualization with seaborn lineplot
    plt.figure(figsize=figsize)

    # Interval to plot
    if start < min(df[time]):
        print(
            "WARNING: Plot start exceeds subset limit. Truncating to subset start date..."
        )
        start = min(df[time])
    if end > max(df[time]):
        print(
            "WARNING: Plot end exceeds subset limit. Truncating to subset end date..."
        )
        end = max(df[time])
    df = df[(df[time] >= start) & (df[time] <= end)]
    df_missing = df_missing[(df_missing[time] >= start) & (df_missing[time] <= end)]

    # Title and subtitle
    if plot_title is None:
        plot_title = f"Time Series sequence for {y_label}"
    if plot_sup_title is None:
        plot_sup_title = (
            f"From {start:%H %p}, {start:%d-%b-%Y} to {end:%H %p}, {end:%d-%b-%Y}"
        )

    # Lineplot
    l1 = sns.lineplot(
        x=time,
        y=value,
        data=df_missing,
        hue=df_missing[value].isna().cumsum(),
        palette=["black"] * sum(df_missing[value].isna()),
        legend=False,
    )
    if missing_only:
        missing_index = np.where(df_missing[value].isnull())[0]
        # Including previous observation
        missing_index = np.insert(missing_index, 0, np.min(missing_index) - 1)
        # Including subsequent observation
        missing_index = np.insert(
            missing_index, missing_index.shape[0], np.max(missing_index) + 1
        )
        plt.plot(
            time, value, data=df.iloc[missing_index], color="orange", label="Missing"
        )
    else:
        plt.plot(time, value, data=df, color="orange", label="Missing")
    # Filling values to indicate NaN
    if fill:
        l1.fill_between(x=time, y1=value, data=df_missing, alpha=0.65)

    l1.set_title(plot_title, y=1.05, fontsize=font_size, fontweight="bold")
    l1.set_ylabel(y_label)
    l1.set_xlabel(x_label)
    plt.legend(loc="upper right", fontsize=font_size)
    plt.suptitle(plot_sup_title, y=0.92, fontsize=font_size)
    plt.xticks(rotation=x_rotate)
    plt.show()
