import dask.dataframe as dd
import dask.array as da

import pandas as pd
import numpy as np


def describe_columns(filepath: str, dtypes: dict) -> dict:
    num_columns = [name for name, dtype in dtypes.items() if dtype == "numerical"]
    cat_columns = [name for name, dtype in dtypes.items() if dtype == "non-numerical"]

    result = dict()

    num_data = dd.read_parquet(filepath, engine="pyarrow", columns=num_columns)
    if not all([pd.api.types.is_numeric_dtype(num_data[col]) for col in num_columns]):
        raise TypeError

    numerical_descriptions = describe_numerical_columns(num_data, num_columns)
    result.update(numerical_descriptions)

    cat_data = dd.read_parquet(filepath, engine="pyarrow", columns=cat_columns)
    if any([pd.api.types.is_numeric_dtype(cat_data[col]) for col in cat_columns]):
        raise TypeError

    categorical_descriptions = describe_categorical_columns(cat_data, cat_columns)
    result.update(categorical_descriptions)

    return result


def describe_numerical_columns(data: dd.DataFrame, columns: list) -> dict:
    result = pd.concat(
        [
            data.describe()
            .compute()
            .rename({"25%": "25", "50%": "50", "75%": "75", "max": "maximum"}),
            pd.DataFrame(data.min().compute(), columns=["minimum"]).T,
            pd.DataFrame(data.isna().sum().compute(), columns=["missing"]).T,
            pd.DataFrame(np.isinf(data).sum().compute(), columns=["infinite"]).T,
            pd.DataFrame((data == 0).sum().compute(), columns=["zeros"]).T,
            pd.DataFrame((data < 0).sum().compute(), columns=["negative"]).T,
        ]
    ).to_dict()
    for col in columns:
        result[col]["bins"] = dask_binning(data, col, 20)
    return result


def describe_categorical_columns(data: dd.DataFrame, columns: list) -> dict:
    count = len(data)
    result = pd.DataFrame(data.isna().sum().compute(), columns=["missing"]).T.to_dict()
    for col in columns:
        result[col]["distinct"] = data[col].nunique().compute()
        result[col]["count"] = count
        if result[col]["distinct"] > 49:
            result[col]["freqs"] = {"error": "TOO MANY VALUES (> 49)"}
            continue
        result[col]["freqs"] = data[col].value_counts().compute().to_dict()
    return result


def dask_binning(dataframe: dd.DataFrame, column: str, bins: int) -> dict:
    bin_size = 100 // bins
    bins_range = np.arange(bin_size, 100 + bin_size, 100 // bins)
    percentiles = np.array(da.percentile(dataframe[column].values, bins_range))
    histogram, _ = da.histogram(dataframe[column], percentiles)
    histogram = np.array(histogram)
    return dict(zip(bins_range, histogram))
