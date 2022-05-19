from typing import Callable, Tuple, Union
from datetime import datetime as dt
import pandas as pd
import numpy as np


def get_y_trues(
    data:pd.core.frame.DataFrame
) -> pd.core.frame.DataFrame:
    """
        Preprocess the data : first col=userId, second col=itemIds(it has all interaction history for user.)
    """
    return data.groupby("userId")["itemId"].unique().reset_index(name="itemIds")


def train_test_split(
    data: pd.core.frame.DataFrame, 
    split_date: Union[int, dt]
) -> Tuple[pd.core.frame.DataFrame]:
    """
        the data split by timestamp
    """

    return data[data["timestamp"] <= split_date], data[data["timestamp"] > split_date]


