import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

    # Interval length
    days_total = (end - start).days + (end - start).seconds // 3600 // 24

    # Lineplot Matplotlib
    fig, ax = plt.subplots(figsize=figsize)
    # > Semester format
    if days_total > 180:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b, %Y"))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    # > Monthly format
    elif days_total > 30:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
    # > Bi-weekly format
    elif days_total > 14:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=12))
    # > 5-days format
    elif days_total > 5:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
    # <= 5-days format
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_locator(mdates.HourLocator())
    ax.plot(time, value, data=subset)
    ax.fill_between(x=time, y1=value, data=subset, alpha=0.65)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title, y=1.05, fontsize=font_size, fontweight="bold")
    plt.suptitle(plot_sup_title, y=0.92, fontsize=font_size - 1)
    fig.autofmt_xdate()
    plt.show()


# Auxiliar function to plot overlaying subsets
def plot_compare(
    df,
    df_missing,
    time="DATE_PST",
    value="RAW_VALUE",
    df_label="Imputed data",
    df_missing_label="Real data",
    missing_range=None,
    start=None,
    end=None,
    y_label="PM 2.5",
    x_label="Date",
    fill=True,
    color="orange",
    lw=2,
    linestyle="dashed",
    missing_only=True,
    plot_title=None,
    plot_sup_title=None,
    font_size=12,
    figsize=(15, 5),
):
    """
    Plot times series subset within passed start and end date interval.
    df: Pandas DataFrame with a time series and a value column.
    df_missing: Pandas DataFrame with a time series of the same length as
    the one in df and a value column.
    time: DateTime variable in the dataset.
    value: Value variable or list of variables to subset with date column.
    df_label: Label for lineplot on df value column.
    df_missing_label: Label for lineplot on df_missing value column.
    missing_range: Optional missing range interval to subset df_missing.
    start: Start date for interval.
    end: End date for interval.
    y_label: Plot label for Y axis.
    x_label: Plot label for X axis.
    color: Color for lineplot on df_missing value column.
    lw: Line width for lineplot on df_missing value column.
    linestyle: Line style for lineplot on df_missing value column.
    fill: Boolean argument to fill area underneath df1 lines.
    missing_only: Boolean arugment to plot only non-missing equivalent of df_missing.
    plot_title: Plot title.
    plot_sup_title: Plot sub-title.
    font_size: Plot tile font size.
    figsize: Plot canvas size.
    """
    sns.set_theme(style="ticks", palette="mako")
    warnings.filterwarnings("ignore", category=UserWarning)

    # Assert time columns of df and df_missing are datetime objects
    assert pd.api.types.is_datetime64_any_dtype(
        df[time]
    ), f"Column {time} of {df} should be of date time format."

    assert pd.api.types.is_datetime64_any_dtype(
        df_missing[time]
    ), f"Column {time} of {df_missing} should be of date time format."

    # Assert sequence of df matches the one of df_missing
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

    # Interval length
    days_total = (end - start).days + (end - start).seconds // 3600 // 24

    # Mainplot with missing values (df_missing)
    # Lineplot Matplotlib
    fig, ax = plt.subplots(figsize=figsize)
    # > Semester format
    if days_total > 180:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b, %Y"))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    # > Monthly format
    elif days_total > 30:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
    # > Bi-weekly format
    elif days_total > 14:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=12))
    # > 5-days format
    elif days_total > 5:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
    # <= 5-days format
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_locator(mdates.HourLocator())
    ax.plot(time, value, data=df_missing, label=df_missing_label)
    if fill:
        ax.fill_between(x=time, y1=value, data=df_missing, alpha=0.65)

    # Secondary plot with complete or imputed values (df)
    if missing_only:
        if missing_range is None:
            missing_index = np.where(df_missing[value].isnull())[0]
            # Including previous observation
            missing_index = np.insert(missing_index, 0, np.min(missing_index) - 1)
            # Including subsequent observation
            missing_index = np.insert(
                missing_index, missing_index.shape[0], np.max(missing_index) + 1
            )
        elif type(missing_range) == type(pd.DataFrame()):
            missing_range = missing_range[
                (missing_range[time] >= start) & (missing_range[time] <= end)
            ]
            missing_index = np.where(missing_range[value].isnull())[0]
            # Including previous observation
            missing_index = np.insert(missing_index, 0, np.min(missing_index) - 1)
            # Including subsequent observation
            missing_index = np.insert(
                missing_index, missing_index.shape[0], np.max(missing_index) + 1
            )
        else:
            missing_index = missing_range
        ax.plot(
            time,
            value,
            data=df.iloc[missing_index],
            color=color,
            lw=lw,
            linestyle=linestyle,
            label=df_label,
        )
    else:
        ax.plot(
            time,
            value,
            data=df,
            color=color,
            lw=lw,
            linestyle=linestyle,
            label=df_label,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title, y=1.05, fontsize=font_size, fontweight="bold")
    plt.legend(loc="upper right", fontsize=font_size)
    plt.suptitle(plot_sup_title, y=0.92, fontsize=font_size - 1)
    fig.autofmt_xdate()
    plt.show()
