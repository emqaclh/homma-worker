import pandas as pd
import numpy as np


def generate_temp_dataset(N: int) -> pd.DataFrame:
    data = pd.DataFrame()
    data["num_var_1"] = np.random.rand(N)
    data["cat_var_1"] = np.random.randint(192, 199, N)
    data["cat_var_1"] = data.cat_var_1.map(chr)
    return data
