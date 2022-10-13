import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import timedelta

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
    TODO: Refactor to consider datetime index!

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
    days_later: Amount of days to subset after the missing_type interval.
    returns: (start_date, end_date) sequence.
    """
    # Assert time column of data is a datetime object
    assert pd.api.types.is_datetime64_any_dtype(
        data[time]
    ), f"Column {time} should be of date time format."
    assert missing_type.upper() in [
        "IMV",
        "CMS",
    ], f"missing_type should be 'IMV' or 'CMS', got {missing_type}."

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
        if (not pd.isnull(start)) and (not pd.isnull(end)):
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
        print(
            f"{len(length)} {missing_type} missing value sequences found with current arguments:"
        )
        print(tabulate(sequences_df, headers="keys", tablefmt="psql"), "\n")

    # Sequence to return (_exp suffix stands for expanded dates)
    start_date = start_list[sequence_no]
    end_date = end_list[sequence_no]
    start_date_exp = start_date - timedelta(days=days_prior)
    end_date_exp = end_date + timedelta(days=days_later)

    try:
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

    # Checking for start and end dates falling outside original dataset time range
    # if start_date_exp < min(subset[time]):
    if start_date_exp < min(data[time]):
        if verbose:
            print(
                f"WARNING: Sequence start exceeds subset limit. Truncating to subset start date..."
            )
        # start_date_exp = min(subset[time]) - timedelta(days=days_prior)
        start_date_exp = min(data[time])
    # if end_date_exp > max(subset[time]):
    if end_date_exp > max(data[time]):
        if verbose:
            print(
                f"WARNING: Sequence end exceeds subset limit. Truncating to subset end date..."
            )
        # end_date_exp = max(subset[time]) + timedelta(days=days_later)
        end_date_exp = max(data[time])

    if verbose:
        print(f"Final interval{(str(start_date_exp), str(end_date_exp))}")

    return (start_date_exp, end_date_exp)


# Auxiliar function to generate artificially missing data
def create_missing(
    data,
    time=None,
    value=None,
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

    # Assert time column of data is a datetime object
    assert pd.api.types.is_datetime64_any_dtype(
        time_index
    ), f"{data} index or time column {time} should be of date time format."

    if start is None:
        start = min(time_index)
    if end is None:
        end = max(time_index)

    assert end > start, f"End date should be higher than start date."
    assert (end - start) < (
        max(time_index) - min(time_index)
    ), f"Sequence length should not exceed subset {time} length."

    # Interval to subset
    if start <= min(time_index):
        print(
            "WARNING: Series' start exceeds subset limit. Truncating to subset start date..."
        )
        start = min(time_index)
    if end >= max(time_index):
        print(
            "WARNING: Series' end exceeds subset limit. Truncating to subset end date..."
        )
        end = max(time_index)
    subset = data[(time_index >= start) & (time_index <= end)]

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