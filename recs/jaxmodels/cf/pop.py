from typing import Optional
import pandas as pd
import numpy as np


class Pop:

    def __init__(
        self,
        top_n:Optional[int]=20,
        itemkey:Optional[str]="itemId"
    ):
        self._top_n = top_n
        self._itemkey = itemkey
    
    def fit(self, df:pd.core.frame.DataFrame):
        self._pop_items = np.array(df[self._itemkey].value_counts().index[:self._top_n])
    
    def predict(self, sessId:Optional[int]=0):
        return self._pop_items
    

class SPop:
    
    def __init__(
        self, 
        top_n:Optional[int]=20, 
        sessionkey:Optional[str]="sessionId",
        itemkey:Optional[str]="itemId"
    ):
        self._top_n = top_n
        self._sessionkey = sessionkey
        self._itemkey = itemkey