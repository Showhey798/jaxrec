import pandas as pd
import numpy as np


class Pop:
    
    def __init__(self, top_n = 100, item_key = 'itemId', support_by_key = None):
        self.top_n = top_n
        self.item_key = item_key
        self.support_by_key = support_by_key
    
    def fit(self, data):
        grp = data.groupby(self.item_key)
        self.pop_list = grp.size() if self.support_by_key is None else grp[self.support_by_key].nunique()
        self.pop_list = self.pop_list / (self.pop_list + 1)
        self.pop_list.sort_values(ascending=False, inplace=True)
        self.pop_list = self.pop_list.head(self.top_n)  
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids):

        preds = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, self.pop_list.index)
        preds[mask] = self.pop_list[predict_for_item_ids[mask]]
        return pd.Series(data=preds, index=predict_for_item_ids)