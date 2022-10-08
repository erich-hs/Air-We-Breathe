import numpy as np

def rmse_score(df, df_missing, value="RAW_VALUE"):
    '''
    df: Pandas DataFrame with a time series and a value column.
    df_missing: Pandas DataFrame with a time series and a value column.
    value: Value variable or list of variables to subset with date column.
    returns: a list of scores
    '''
    rmse = np.sqrt(np.mean((df[value] - df_missing[value])**2))
    print(f"RMSE for {value}: {rmse:.4f}")
    return rmse