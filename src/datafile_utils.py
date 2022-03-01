import shutil

import pandas as pd
import dask.dataframe as dd

import os


def validate_csv_setup(filepath: str, encoding: str, sep: chr, decimal: chr) -> dict:
    setup = dict(encoding=encoding, sep=sep, decimal=decimal)

    try:
        columns = pd.read_csv(
            filepath, nrows=1, **setup
        ).columns  ## evaluate base columns
        dtypes = dict(
            [(col, "numerical") for col in columns]
        )  ## initialize base dtypes

        for chunk in pd.read_csv(
            filepath, chunksize=10_000, **setup
        ):  ## lazy loading by chunks to avoid excessive RAM usage
            columns = [
                col for col in chunk.columns if dtypes[col] == "numerical"
            ]  ## consider only (still-)numerical columns
            for col in columns:
                if not pd.api.types.is_numeric_dtype(chunk[col]):
                    dtypes[col] = "non-numerical"

            if all(map(lambda x: x == "non-numerical", dtypes.values())):
                break  ## if there is no numerical columns left, stop

        return dict(result=dtypes, setup=setup)

    except Exception as error:
        return dict(error=error.__class__.__name__)


def consolidate_datafile(
    filepath: str, output: str, setup: dict = {}, temp_folder: str = ""
) -> dict:

    try:
        os.makedirs(temp_folder, exist_ok=True)

        if not all([key in setup.keys() for key in ("encoding", "sep", "decimal")]):
            raise  ## check if setup keys were provided

        for i, chunk in enumerate(pd.read_csv(filepath, chunksize=50_000, **setup)):
            chunk.to_csv(f"{temp_folder}/temp_{i}.csv", index=False)

        dd.read_csv(f"{temp_folder}/temp_*.csv").to_parquet(
            f"{output}.parquet", engine="pyarrow"
        )

        shutil.rmtree(temp_folder)

        return dict(done=True)

    except Exception as error:
        return dict(done=False, error=error.__class__.__name__)
