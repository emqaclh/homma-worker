import pytest

import os

import dask.dataframe as dd

from src.feature_utils import *

from .utils import generate_temp_dataset


class TestFeatureDescription:
    @pytest.fixture(scope="session")
    def temp_parquet(self, tmp_path_factory):
        data = generate_temp_dataset(150_000)
        base_path = tmp_path_factory.mktemp("temp")
        csv_path = str(base_path / "test.csv")
        parquet_path = str(base_path / "test.parquet")
        data.to_csv(csv_path, encoding="latin-1", sep=";", decimal=",", index=False)
        dd.read_csv(csv_path, encoding="latin-1", sep=";", decimal=",").to_parquet(
            parquet_path, engine="pyarrow"
        )
        os.remove(csv_path)
        return parquet_path

    def test_feature_description(self, temp_parquet):
        result = describe_columns(
            temp_parquet, dict(num_var_1="numerical", cat_var_1="non-numerical")
        )
        for key in (
            "missing",
            "infinite",
            "mean",
            "std",
            "25",
            "50",
            "75",
            "minimum",
            "maximum",
            "zeros",
            "negative",
            "bins",
            "count",
        ):
            assert key in result["num_var_1"].keys()
        for key in ("distinct", "missing", "freqs", "count"):
            assert key in result["cat_var_1"].keys()

    def test_feature_description_wrong_dtypes(self, temp_parquet):
        with pytest.raises(TypeError):
            describe_columns(
                temp_parquet, dict(num_var_1="non-numerical", cat_var_1="numerical")
            )

    def test_feature_description_no_file(self):
        with pytest.raises(FileNotFoundError):
            describe_columns(
                "test.txt", dict(num_var_1="numerical", cat_var_1="non-numerical")
            )
