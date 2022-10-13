import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

def rmse_score(data, data_missing, value="RAW_VALUE"):
    """
    data: Pandas DataFrame with a time series and a value column.
    data_missing: Pandas DataFrame with a time series and a value column.
    value: Value variable or list of variables to subset with date column.
    returns: a list of scores
    """
    rmse = np.sqrt(np.mean((data[value] - data_missing[value]) ** 2))
    print(f"RMSE for {value}: {rmse:.4f}")
    return rmse

def stationarity_test(data,
                      value,
                      time=None,
                      start=None,
                      end=None,
                      fillna="interpolate",
                      dropna=True,
                      autolag="AIC"):
    '''
    Perform and return results for "ADF" and "KPSS" stationarity tests.

    Augmented Dickey-Fuller (ADF) test hypothesis:
    H0: A unit root is present in the time series sample (Non-stationary)
    Ha: There is no root unit present in the time series sample (Stationary)

    Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test hypothesis:
    H0: The data is stationary around a constant (Stationary)
    Ha: A unit root is present in the time series sample (Non-stationary)

    output: Output dict of a statsmodels adfuller or kpss test.
    test: Either 'adf' or 'kpss' to define decision based on p-value.

    data: Pandas DataFrame with a time series and a value column.
    start: Start date for interval.
    end: End date for interval.
    fillna: ["interpolate", "ffill", "bfill"] method to fill missing values.
    dropna: Boolean to indicate whether to drop missing values not successfully filled with fillna.
    autolag: [“AIC”, “BIC”, “t-stat”, None] method to use for automatically determining
    lag length amount for the ADF test.
    returns: Pandas DataFrame with KPSS and ADF test results.
    '''
    if type(data.index) == pd.core.indexes.datetimes.DatetimeIndex:
        time_index = data.index
    else:
        try:
            min(data[time])
        except KeyError:
            print(f"Dataframe index is not a DateTime object. Please specify a valid column for argument time.")
        time_index = data[time]

    if start is None:
        start = min(time_index)
    if end is None:
        end = max(time_index)
    
    # Subset
    if start < min(time_index):
        print(
            "WARNING: Interval start exceeds input data limit. Truncating to data's start date..."
        )
        start = min(time_index)
    if end > max(time_index):
        print(
            "WARNING: Interval end exceeds input data limit. Truncating to data's end date..."
        )
        end = max(time_index)
    subset = data[(time_index >= start) & (time_index <= end)][value]

    if fillna.lower() == "interpolate":
        subset = subset.interpolate()
    elif fillna.lower() == "ffill":
        subset = subset.ffill()
    elif fillna.lower() == "bfill":
        subset = subset.bfill()
    else:
        return f"Invalid fillna parameter. Expected one of interpolate, ffill, bfill, got {fillna}."

    if dropna:
        subset = subset.dropna()

    adf_output=adfuller(subset, autolag=autolag)
    kpss_output=kpss(subset)

    def print_results(output, test='adf'):
        '''
        Print results for either "ADF" or "KPSS" tests.
        '''
        pval = output[1]
        test_score = output[0]
        lags = output[2]
        decision = 'Non-Stationary'
        if test == 'adf':
            if pval < 0.05:
                decision = 'Stationary'
        elif test=='kpss':
            if pval >= 0.05:
                decision = 'Stationary'
        output_dict = {
        'Test Statistic': round(test_score, 4),
        'p-value': round(pval, 4),
        'Lags Used': lags,
        'Decision': decision
    }
        return pd.Series(output_dict, name=test)

    results_df=pd.concat([
        print_results(adf_output), print_results(kpss_output, test="kpss")
    ], axis=1)

    return results_df