import pytest

from src.datafile_utils import *

from .utils import generate_temp_dataset


class TestSetupValidation:
    @pytest.fixture(scope="session")
    def temp_csv(self, tmp_path_factory):
        data = generate_temp_dataset(150_000)
        path = tmp_path_factory.mktemp("temp") / "test.csv"
        path = str(path)
        data.to_csv(path, encoding="latin-1", sep=";", decimal=",", index=False)
        return path

    def test_validate_csv_valid_setup(self, temp_csv):
        validation = validate_csv_setup(temp_csv, "latin-1", sep=";", decimal=",")
        assert "result" in validation.keys()
        assert validation["result"]["cat_var_1"] == "non-numerical"
        assert validation["result"]["num_var_1"] == "numerical"

    def test_validate_csv_nonexistent_file(self):
        validation = validate_csv_setup("test.txt", "latin-1", sep=";", decimal=",")
        assert "error" in validation.keys()

    def test_validate_csv_wrong_encoding(self, temp_csv):
        validation = validate_csv_setup(temp_csv, "XXX", sep=";", decimal=",")
        assert "error" in validation.keys()

    def test_validate_csv_wrong_sep(self, temp_csv):
        validation = validate_csv_setup(temp_csv, "latin-1", sep="|", decimal=",")
        assert len(validation["result"]) != 2

    def test_validate_csv_wrong_decimal(self, temp_csv):
        validation = validate_csv_setup(temp_csv, "latin-1", sep=";", decimal=".")
        assert validation["result"]["num_var_1"] == "non-numerical"

    def test_validate_csv_missing_params(self, temp_csv):
        with pytest.raises(Exception):
            validation = validate_csv_setup(temp_csv, "latin-1")
