from typing import Optional, Callable, Tuple, Union, Dict, Any
import os
from datetime import datetime as dt
import pandas as pd
import numpy as np
import pickle


def get_y_trues(
    data:pd.core.frame.DataFrame,
    sessionkey:Optional[str]="userId",
    itemkey:Optional[str]="itemId"    
) -> pd.core.frame.DataFrame:
    """
        Preprocess the data : first col=userId, second col=itemIds(it has all interaction history for user.)
    """
    return data.groupby(sessionkey)[itemkey].unique().reset_index(name="itemIds")


def train_test_split(
    data: pd.core.frame.DataFrame, 
    split_date: Union[int, dt]
) -> Tuple[pd.core.frame.DataFrame]:
    """
        the data split by timestamp
    """

    return data[data["timestamp"] <= split_date], data[data["timestamp"] > split_date]

def save_params(
    params: Dict[str, Any],
    name:Optional[str]="model",
    path : Optional[str]= "~/work/recs/params/"
):
    save_path = os.path.join(path, name)
    print("=====saving parameters=====")
    pickle.dump(params, save_path)
    print("=====done=====")
    
    


