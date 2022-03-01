import pytest

import dask.dataframe as dd

from src.datafile_utils import *

from .utils import generate_temp_dataset


class TestDatasetConsolidation:
    @pytest.fixture(scope="session")
    def temp_csv(self, tmp_path_factory):
        data = generate_temp_dataset(150_000)
        path = tmp_path_factory.mktemp("temp") / "test.csv"
        path = str(path)
        data.to_csv(path, encoding="latin-1", sep=";", decimal=",", index=False)
        return path

    def test_csv_consolidation_valid(self, temp_csv):
        consolidation = consolidate_datafile(
            temp_csv,
            "/home/emqaclh/dev/homma/files/test",
            dict(encoding="latin-1", sep=";", decimal=","),
            "/home/emqaclh/dev/homma/files/temp",
        )
        output = dd.read_parquet(
            "/home/emqaclh/dev/homma/files/test.parquet", engine="pyarrow"
        )
        assert consolidation["done"]
        assert isinstance(output, dd.DataFrame)
        shutil.rmtree("/home/emqaclh/dev/homma/files/test.parquet")

    def test_csv_consolidation_nonexistent_file(self):
        consolidation = consolidate_datafile(
            "test.txt",
            "/home/emqaclh/dev/homma/files/test",
            dict(encoding="latin-1", sep=";", decimal=","),
            "/home/emqaclh/dev/homma/files/temp",
        )
        assert not consolidation["done"]

    def test_csv_consolidation_empty_setup(self, temp_csv):
        consolidation = consolidate_datafile(
            temp_csv,
            "/home/emqaclh/dev/homma/files/test",
            {},
            "/home/emqaclh/dev/homma/files/temp",
        )
        assert not consolidation["done"]

    def test_csv_consolidation_wrong_setup(self, temp_csv):
        consolidation = consolidate_datafile(
            temp_csv,
            "/home/emqaclh/dev/homma/files/test",
            dict(encoding="XXX", sep=";", decimal=","),
            "/home/emqaclh/dev/homma/files/temp",
        )
        assert not consolidation["done"]

    def test_csv_consolidation_no_temp_folder(self, temp_csv):
        consolidation = consolidate_datafile(
            temp_csv,
            "/home/emqaclh/dev/homma/files/test",
            dict(encoding="latin-1", sep=";", decimal=","),
            "/home/emqaclh/dev/homma/files/nonexistent",
        )
        assert consolidation["done"]
        shutil.rmtree("/home/emqaclh/dev/homma/files/test.parquet")
