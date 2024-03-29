from typing import List, Optional

import pandas as pd
import numpy as np
from .metrics import precision_at_k, recall_at_k, ndcg_at_k, mrr_at_k, map_at_k

def evaluate(
    df:pd.core.frame.DataFrame,
    k: Optional[int] = 5,
    metrics: Optional[List[str]] = ["precision", "map", "recall", "ndcg", "mrr"]
):
    """
        y_truesとy_predsは各ユーザに対してアイテムidの集合をもつデータフレームとする

        ex)
        userId | itemIds
        0   | [123, 32, 24, ...]
        .
        .
        .
    """

    #df = pd.merge(y_trues, y_preds, on="userId")
    
    eval_scores = {}
    for metric in metrics:
        
        if metric == "precision":
            eval_scores[metric] = df[["trueIds", "predIds"]].apply(lambda x: precision_at_k(x[0], x[1], k), axis=1)
        if metric == "map":    
            eval_scores["map"] = df[["trueIds", "predIds"]].apply(lambda x: map_at_k(x[0], x[1], k), axis=1)
        
        if metric == "recall":
            eval_scores[metric] = df[["trueIds", "predIds"]].apply(lambda x: recall_at_k(x[0], x[1], k), axis=1)

        if metric == "ndcg":
            eval_scores[metric] = df[["trueIds", "predIds"]].apply(lambda x: ndcg_at_k(x[0], x[1], k), axis=1)
        
        if metric == "mrr":
            eval_scores[metric] = df[["trueIds", "predIds"]].apply(lambda x: mrr_at_k(x[0], x[1], k), axis=1)
      
    eval_metric_df = pd.concat([eval_scores[key] for key in eval_scores.keys()], axis=1)

    eval_metric_df.columns = metrics
        
    return eval_metric_df, eval_metric_df.mean()