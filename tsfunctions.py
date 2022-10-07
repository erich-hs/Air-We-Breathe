# Auxiliar function to subset around missing intervals
def subset_interval(
    data,
    time="DATE_PST",
    value="RAW_VALUE",
    column_type_missing="MISSING_SAMPLE",
    column_missing="MISSING",
    column_missing_sequence="MISSING_SEQ",
    missing_type=None,
    sequence_no=0,
    days_prior=5,
    days_later=5,
    verbose=True,
):
    """
    Subset timeseries interval around an Individual Missing Value (IMV) or
    a Continuous Missing Sample (CMS):
    data: Pandas DataFrame with a time series and a value column.
    time: DateTime variable in the dataset.
    value: Value variable or list of variables to subset with date column.
    column_type_missing: Variable that stores missing value type.
    column_missing: Dummy variable that stores missing value indicator.
    column_missing_sequence: Variable that stores length of current missing sequence.
    missing_type: Type of missing value to search for. Either 'IMV' or 'CMS'.
    sequence_no: Which missing interval to use in the sequence of missing_type found.
    days_prior: Amount of days to subset prior to the missing_type interval.
    days_later: Amount of days to subsat after the missing_type interval.
    returns: (start_date, end_date) sequence.
    """
    # Subsetting to missing values
    if missing_type == "IMV":
        subset = data[data[column_type_missing] == 0]
        subset = subset[subset[column_missing] == 1]
    elif missing_type == "CMS":
        subset = data[data[column_type_missing] == 1]
    else:
        return (
            "Invalid or non-defined value type! Please select between: 'IMV' or 'CMS'"
        )

    # Initializing start and end intervals, indices, star, and end lists
    start = min(subset[time])
    end = 0
    indices = []
    start_list = []
    end_list = []

    # Defining time deltas
    for i, row in subset.iterrows():
        try:
            td = i - indices[-1]
        except IndexError:
            # Initialize to 0 at the beginning of the subset iteration
            td = 0
        indices.append(i)

        # Set end date for the first missing sequence in the subset
        if not end:
            end = start + timedelta(hours=row[column_missing_sequence])

        # Storing start and end date when time delta > 1
        if td > 1:
            start = row[time]
            end = start + timedelta(hours=row[column_missing_sequence])

        # Appending start and end date for newly found sequences
        if start not in start_list:
            start_list.append(start)
        if end not in end_list:
            end_list.append(end)

    # Printing start and end lists for sequence sizes
    length = []
    for end, start in zip(end_list, start_list):
        length.append(
            str(int((end - start).days * 24 + (end - start).seconds / 3600)) + " h"
        )
    sequences_df = pd.DataFrame(
        {"Sequence Start": start_list, "Sequence End": end_list, "Length": length}
    )
    if verbose:
        print("Missing value sequences found with current argument:")
        print(tabulate(sequences_df, headers="keys", tablefmt="psql"), "\n")

    # Sequence to return
    start_date = start_list[sequence_no]
    end_date = end_list[sequence_no]
    start_date_exp = start_date - timedelta(days=days_prior)
    end_date_exp = end_date + timedelta(days=days_later)

    # Checking for end date falling within a missing sequence
    if (end_date_exp >= start_list[sequence_no + 1]) and (
        end_date_exp <= end_list[sequence_no + 1]
    ):
        if verbose:
            print(
                "WARNING: Sequence ending on a missing interval. Expanding to incorporate next sequence..."
            )
        end_date = end_list[sequence_no + 1]
        end_date_exp = end_date + timedelta(days=days_later)

    # Checking for start date falling within a missing sequence
    try:
        if (start_date_exp >= start_list[sequence_no - 1]) and (
            start_date_exp <= end_list[sequence_no - 1]
        ):
            if verbose:
                print(
                    "WARNING: Sequence starting on a missing interval. Expanding to incorporate previous sequence..."
                )
            start_date = start_list[sequence_no - 1]
            start_date_exp = start_date - timedelta(days=days_prior)
    except IndexError:
        pass

    else:
        if verbose:
            print(
                f"Returning interval for #{sequence_no} matched {missing_type} sequence: {(str(start_date), str(end_date))}"
            )
            missing_tot = sum(
                subset[value][
                    (subset[time] >= start_date_exp) & (subset[time] <= end_date_exp)
                ].isna()
            )
            print(f"Sum of {missing_type} sequence(s) length(s): {missing_tot} hours")

    if start_date_exp < min(subset[time]):
        if verbose:
            print(
                f"WARNING: Sequence start exceeds subset limit. Truncating to subset start date..."
            )
        start_date_exp = min(subset[time]) - timedelta(days=days_prior)
    if end_date_exp > max(subset[time]):
        if verbose:
            print(
                f"WARNING: Sequence end exceeds subset limit. Truncating to subset end date..."
            )
        end_date_exp = max(subset[time]) + timedelta(days=days_later)

    if verbose:
        print(f"Final interval{(str(start_date_exp), str(end_date_exp))}")

    return (start_date_exp, end_date_exp)


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


# Auxiliar function to generate artificially missing data
def create_missing(
    data,
    time="DATE_PST",
    value="RAW_VALUE",
    start=None,
    end=None,
    missing_length=1,
    missing_index="end",
    padding=24,
):
    """
    data: Pandas DataFrame with a time series and a value column.
    time: DateTime variable in the dataset.
    value: Value variable or list of variables to subset with date column.
    start: Start date for interval.
    end: End date for interval.
    missing_length: Length of missing sample sequence.
    missing_index: Index where missing sample will be generated at. Can take
    any of the following string arguments ['start', 'end'].
    padding: padding to shift generated missing interval from 'start' or 'end'.
    returns: subset, subset_missing, where df is the subset of original data within
    start and end arguments, and df_missing is a copy of subset with missing
    data.
    """
    if start is None:
        start = min(data[time])
    if end is None:
        end = max(data[time])

    # Interval to subset
    if start <= min(data[time]):
        print(
            "WARNING: Series' start exceeds subset limit. Truncating to subset start date..."
        )
        start = min(data[time])
    if end >= max(data[time]):
        print(
            "WARNING: Series' end exceeds subset limit. Truncating to subset end date..."
        )
        end = max(data[time])
    subset = data[(data[time] >= start) & (data[time] <= end)]

    # Looking for missing value on original subset
    tot_missing = subset[value].isna().sum()
    if tot_missing:
        print(
            f"Original subset contains missing data. Total missing values {tot_missing}."
        )

    # Missing data subset
    subset_missing = subset.copy()
    subset_array = subset_missing[value].to_numpy()
    if missing_index == "end":
        missing_start = padding + missing_length
        missing_end = padding
        subset_array[-missing_start:-missing_end] = np.NaN
    elif missing_index == "start":
        missing_start = padding
        missing_end = padding + missing_length
        subset_array[missing_start:missing_end] = np.NaN
    subset_missing[value] = subset_array

    return subset, subset_missing


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
    df1: Pandas DataFrame with a time series and a value column.
    df2: Pandas DataFrame with a time series and a value column.
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
    l1 = sns.lineplot(x=time, y=value, data=df_missing,
                     hue=df_missing[value].isna().cumsum(),
                     palette=["black"]*sum(df_missing[value].isna()),
                     legend=False)
    if missing_only:
        missing_index = np.where(df_missing[value].isnull())[0]
        # Including previous observation
        missing_index = np.insert(missing_index, 0, np.min(missing_index)-1)
        # Including subsequent observation
        missing_index = np.insert(missing_index, missing_index.shape[0], np.max(missing_index)+1)
        plt.plot(time, value, data = df.iloc[missing_index], color = 'orange', label = "Missing")
    else:
        plt.plot(time, value, data = df, color = 'orange', label = "Missing")
    # Filling values to indicate NaN
    if fill:
        l1.fill_between(x=time, y1=value, data=df_missing, alpha=0.65)

    l1.set_title(plot_title, y=1.05, fontsize=font_size, fontweight="bold")
    l1.set_ylabel(y_label)
    l1.set_xlabel(x_label)
    plt.legend(loc = "upper right", fontsize=font_size)
    plt.suptitle(plot_sup_title, y=0.92, fontsize=font_size)
    plt.xticks(rotation=x_rotate)
    plt.show()