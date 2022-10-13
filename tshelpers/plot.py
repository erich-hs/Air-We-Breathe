import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# Auxiliar function to plot a subset
def plot_sequence(
    data,
    time=None,
    value=None,
    start=None,
    end=None,
    y_label="PM 2.5",
    x_label="Date",
    fill=True,
    plot_title=None,
    plot_sup_title=None,
    font_size=12,
    figsize=(15, 5),
    palette="mako",
    style="ticks"
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
    fill: Boolean argument to fill area underneath data lines.
    plot_title: Plot title.
    plot_sup_title: Plot sub-title.
    font_size: Plot tile font size.
    figsize: Plot canvas size.
    palette: Seaborn color palette.
    style: Seaborn plot style.
    """
    sns.set_theme(style=style, palette=palette)

    if type(data.index) == pd.core.indexes.datetimes.DatetimeIndex:
        time_index = data.index
        time_is_index = 1
    else:
        try:
            min(data[time])
        except KeyError:
            print(f"Dataframe index is not a DateTime object. Please specify a valid column for argument time.")
        time_index = data[time]
        time_is_index = 0

    if start is None:
        start = min(time_index)
    if end is None:
        end = max(time_index)

    # Interval to plot
    if start < min(time_index):
        print(
            "WARNING: Plot start exceeds subset limit. Truncating to subset start date..."
        )
        start = min(time_index)
    if end > max(time_index):
        print(
            "WARNING: Plot end exceeds subset limit. Truncating to subset end date..."
        )
        end = max(time_index)
    subset = data[(time_index >= start) & (time_index <= end)]

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
    elif days_total > 35:
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
    # > 1-day format
    elif days_total > 1:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_locator(mdates.HourLocator())
    # <= 1-day format
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H %p"))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
    if time_is_index:
        ax.plot(value, data=subset)
        if fill:
            ax.fill_between(x=subset.index, y1=value, data=subset, alpha=0.65)
    else:
        ax.plot(time, value, data=subset)
        if fill:
            ax.fill_between(x=time, y1=value, data=subset, alpha=0.65)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title, y=1.05, fontsize=font_size, fontweight="bold")
    plt.suptitle(plot_sup_title, y=0.92, fontsize=font_size - 1)
    fig.autofmt_xdate()
    sns.despine()
    plt.show()


# Auxiliar function to plot overlaying subsets
def plot_compare(
    data,
    data_missing,
    time=None,
    value=None,
    value_missing=None,
    data_label="Imputed data",
    data_missing_label="Real data",
    missing_range=None,
    start=None,
    end=None,
    y_label="PM 2.5",
    x_label="Date",
    fill=True,
    color="orange",
    lw=2,
    linestyle="dashed",
    missing_only=False,
    plot_title=None,
    plot_sup_title=None,
    font_size=12,
    figsize=(15, 5),
):
    """
    TODO:
    Auto-assign value and value_missing if data and data_missing have a single column.
    Assert that data_missing has missing values when missing_only=True.

    Plot times series subset within passed start and end date interval.
    data: Pandas DataFrame with a time series and a value column.
    data_missing: Pandas DataFrame with a time series of the same length as
    the one in data and a value column.
    time: DateTime variable in the dataset.
    value: Value variable or list of variables to subset with date column.
    data_label: Label for lineplot on data value column.
    data_missing_label: Label for lineplot on data_missing value column.
    missing_range: Optional missing range interval to subset data_missing.
    start: Start date for interval.
    end: End date for interval.
    y_label: Plot label for Y axis.
    x_label: Plot label for X axis.
    color: Color for lineplot on data_missing value column.
    lw: Line width for lineplot on data_missing value column.
    linestyle: Line style for lineplot on data_missing value column.
    fill: Boolean argument to fill area underneath data lines.
    missing_only: Boolean arugment to plot only non-missing equivalent of data_missing.
    plot_title: Plot title.
    plot_sup_title: Plot sub-title.
    font_size: Plot tile font size.
    figsize: Plot canvas size.
    """
    sns.set_theme(style="ticks", palette="mako")
    warnings.filterwarnings("ignore", category=UserWarning)

    if type(data.index) == pd.core.indexes.datetimes.DatetimeIndex:
        time_index = data.index
        time_missing_index = data_missing.index
        time_is_index = 1
    else:
        try:
            min(data[time])
        except KeyError:
            print(f"Dataframe index is not a DateTime object. Please specify a valid column for argument time.")
        time_index = data[time]
        time_missing_index = data_missing[time]
        time_is_index = 0

    # Assert time columns of data and data_missing are datetime objects
    assert pd.api.types.is_datetime64_any_dtype(
        time_index
    ), f"{data} index or time column {time} should be of date time format."

    assert pd.api.types.is_datetime64_any_dtype(
        time_missing_index
    ), f"{data_missing} index or time column {time} should be of date time format."

    # Assert sequence of df matches the one of df_missing
    assert len(data) == len(data_missing), "Sequences must be of the same length."
    assert min(time_index) == min(
        time_missing_index
    ), "Sequences do not start at the same time stamp."
    assert max(time_index) == max(
        time_missing_index
    ), "Sequences do not end at the same time stamp."

    if start is None:
        start = min(time_index)
    if end is None:
        end = max(time_index)

    # Interval to plot
    if start < min(time_index):
        print(
            "WARNING: Plot start exceeds subset limit. Truncating to subset start date..."
        )
        start = min(time_index)
    if end > max(time_index):
        print(
            "WARNING: Plot end exceeds subset limit. Truncating to subset end date..."
        )
        end = max(time_index)
    data = data[(time_index >= start) & (time_index <= end)]
    data_missing = data_missing[(time_missing_index >= start) & (time_missing_index <= end)]

    # Title and subtitle
    if plot_title is None:
        plot_title = f"Time Series sequence for {y_label}"
    if plot_sup_title is None:
        plot_sup_title = (
            f"From {start:%H %p}, {start:%d-%b-%Y} to {end:%H %p}, {end:%d-%b-%Y}"
        )

    # Interval length
    days_total = (end - start).days + (end - start).seconds // 3600 // 24

    # Mainplot with missing values (data_missing)
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
    # > 1-day format
    elif days_total > 1:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_locator(mdates.HourLocator())
    # <= 1-day format
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H %p"))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
    if time_is_index:
        ax.plot(value_missing, data=data_missing, label=data_missing_label)
        if fill:
            ax.fill_between(x=data_missing.index, y1=value_missing, data=data_missing, alpha=0.65)
    else:
        ax.plot(time, value_missing, data=data_missing, label=data_missing_label)
        if fill:
            ax.fill_between(x=time, y1=value_missing, data=data_missing, alpha=0.65)

    # Secondary plot with complete or imputed values (data)
    if missing_only:
        if missing_range is None:
            missing_index = np.where(data_missing[value_missing].isnull())[0]
            # Including previous observation
            missing_index = np.insert(missing_index, 0, np.min(missing_index) - 1)
            # Including subsequent observation
            missing_index = np.insert(
                missing_index, missing_index.shape[0], np.max(missing_index) + 1
            )
        elif type(missing_range) == type(pd.DataFrame()):
            # Check if missing_range df has a datetime index
            if type(missing_range.index) == pd.core.indexes.datetimes.DatetimeIndex:
                missing_range_index = missing_range.index
            else:
                missing_range_index = missing_range[time]
            missing_range = missing_range[
                (missing_range_index >= start) & (missing_range_index <= end)
            ]
            missing_index = np.where(missing_range[value_missing].isnull())[0]
            # Including previous observation
            missing_index = np.insert(missing_index, 0, np.min(missing_index) - 1)
            # Including subsequent observation
            missing_index = np.insert(
                missing_index, missing_index.shape[0], np.max(missing_index) + 1
            )
        else:
            missing_index = missing_range
        if time_is_index:
            ax.plot(
                value,
                data=data.iloc[missing_index],
                color=color,
                lw=lw,
                linestyle=linestyle,
                label=data_label,
            )
        else:
            ax.plot(
                time,
                value,
                data=data.iloc[missing_index],
                color=color,
                lw=lw,
                linestyle=linestyle,
                label=data_label,
            )
    else:
        if time_is_index:
            ax.plot(
                value,
                data=data,
                color=color,
                lw=lw,
                linestyle=linestyle,
                label=data_label,
            )
        else:
            ax.plot(
                time,
                value,
                data=data,
                color=color,
                lw=lw,
                linestyle=linestyle,
                label=data_label,
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title, y=1.05, fontsize=font_size, fontweight="bold")
    plt.legend(loc="upper right", fontsize=font_size)
    plt.suptitle(plot_sup_title, y=0.92, fontsize=font_size - 1)
    fig.autofmt_xdate()
    plt.show()

def plot_missing(data,
                 start=None,
                 end=None,
                 plot_title=None,
                 plot_sup_title=None,
                 annotate_missing=True,
                 font_size=12,
                 figsize=(12, 3)):
    '''
    data: Pandas DataFrame with a time series and a value column.
    start: Start date for interval.
    end: End date for interval.
    plot_title: Plot title.
    plot_sup_title: Plot sub-title.
    annotate_missing: Boolean argument to specify whether to annotate missing values.
    font_size: Plot tile font size.
    figsize: Plot canvas size.
    '''

    x_tick = 16

    if start is None:
        start = min(data.index)
    if end is None:
        end = max(data.index)

    # Title and subtitle
    if plot_title is None:
        plot_title = f"PM 2.5 Missing Values Heatmap"
    if plot_sup_title is None:
        plot_sup_title = (
            f"From {start:%H %p}, {start:%d-%b-%Y} to {end:%H %p}, {end:%d-%b-%Y}"
        )

    # Interval to plot
    if start < min(data.index):
        print(
            "WARNING: Plot start exceeds subset limit. Truncating to subset start date..."
        )
        start = min(data.index)
    if end > max(data.index):
        print(
            "WARNING: Plot end exceeds subset limit. Truncating to subset end date..."
        )
        end = max(data.index)
    data = data[(data.index >= start) & (data.index <= end)]

    # Tick Labels
    if annotate_missing:
        ytick_labels=[station[:-5].replace("_", " ") + f"\nMissing: {round(data[station].isna().sum()/data[station].count()*100, 2)}%" for station in data.columns]
    else:
        ytick_labels=[station[:-5].replace("_", " ") for station in data.columns]


    fig, ax = plt.subplots()
    sns.heatmap(data.isnull().T, cbar=False)
    ax.figure.set_size_inches(figsize)
    ax.xaxis.set_major_locator(ticker.LinearLocator(x_tick))
    ax.set_xticklabels(
        [
            timestamp.strftime("%d %b, %Y")
            for timestamp in pd.date_range(
                start=start,
                end=end,
                periods=len(ax.get_xticks()),
            ).to_list()
        ],
        fontsize=font_size - 2,
    )
    ax.set_yticklabels(ytick_labels, fontsize=font_size - 2)
    ax.set_xlabel("Date", fontsize=font_size - 1, fontweight="bold")
    ax.set_ylabel("Stations", fontsize=font_size - 1, fontweight="bold")
    ax.set_title(plot_title, y=1.08, fontsize=font_size, fontweight="bold")
    plt.suptitle(plot_sup_title, y=0.94, fontsize=font_size - 1)
    fig.autofmt_xdate()
    plt.show()

def plot_multiple_pacf(data,
                       columns,
                       lags=15,
                       start=None,
                       end=None,
                       plot_title=None,
                       x_label=None,
                       figsize=(9, 9)):
    '''
    Plot Partial Autocorrelation PACF (left) and Autocorrelation ACF (right) estimates
    for data over passed columns.
    data: Pandas DataFrame with a time series and a value column.
    columns: Columns with value variables to plot.
    lags: Number of lags to plot PACF and ACF estimates.
    start: Start date for interval.
    end: End date for interval.
    plot_title: Plot title.
    x_label: X Axis plot label.
    figsize. Plot canvas size.
    '''
    if plot_title == None:
        plot_title="PACF (left) and ACF (right) Correlograms"
    if x_label == None:
        x_label="Lags"
    fig, axs = plt.subplots(len(columns), 2, sharex=True, sharey=True, figsize=figsize)
    for i, column in enumerate(data[columns]):
        for j in [0, 1]:
            args = {
            "x": data[start:end][column],
            "lags": lags,
            "method": "yw",
            "ax": axs[i, j],
            "title": "",
            "zero": False
            }
            if j == 0:
                # Partial Autocorrelation Function plot
                p = plot_pacf(**args)
                title = f"{column.replace('_', ' ') + ' [PACF]'}"
            elif j == 1:
                # Autocorrelation Function plot without "method" parameter
                p = plot_acf(**{parameter:args[parameter] for parameter in args if parameter!='method'})
                title = f"{column.replace('_', ' ') + ' [ACF]'}"
            axs[i, j].xaxis.set_major_locator(ticker.MultipleLocator(1))
            axs[i, j].tick_params(labelsize=9)
            axs[i, j].set_title(title, fontsize=10, y=0.96)
            sns.despine()
            p.set_tight_layout(1)
            if i == len(data[columns].columns) - 1:
                axs[i, j].set_xlabel(x_label, fontsize=10)
    plt.suptitle(plot_title,
                y=0.97,
                fontsize=10,
                fontweight="bold")
    plt.xlim([0, lags+0.5])
    plt.show()


def plot_correlation(data,
                     columns,
                     plot_title=None):
    '''
    Plot a correlation matrix with pearson correlation values on top diagonal and
    correlation scatterplox on bottom diagonal.
    data: Pandas Dataframe with numeric columns.
    columns: Columns to plot correlation from data.
    plot_title: Plot title.
    '''
    # Plot Title
    if plot_title is None:
        plot_title = "Correlation Matrix"

    # Correlation dots
    def corrdot(*args, **kwargs):
        corr_r = args[0].corr(args[1], 'pearson')
        corr_text = f"{corr_r:2.2f}".replace("0.", ".")
        ax = plt.gca()
        ax.set_axis_off()
        marker_size = abs(corr_r) * 7000
        ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="mako",
                vmin=-1, vmax=1, transform=ax.transAxes)
        font_size = abs(corr_r) * 30
        ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                    ha='center', va='center', fontsize=font_size)

    # PairGrid plot
    sns.set_theme(style="ticks", palette="mako")
    g = sns.PairGrid(data[columns],
                    aspect=1.5,                 
                    diag_sharey=False)
    g.map_lower(sns.regplot, marker="+", lowess=True, ci=False, line_kws={'color':'darkblue'})
    g.map_diag(sns.distplot, kde_kws={'color':'darkblue'})
    g.map_upper(corrdot)
    plt.suptitle(plot_title, fontsize=16, fontweight="bold")
    plt.show()