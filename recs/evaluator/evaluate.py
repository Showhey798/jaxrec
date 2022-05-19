from typing import List, Optional

import pandas as pd
from .metrics import precision_at_k, recall_at_k, ndcg_at_k, mrr_at_k, map_at_k

def evaluate(
    y_trues: pd.core.frame.DataFrame,
    y_preds: pd.core.frame.DataFrame,
    k: Optional[int] = 5,
    metrics: Optional[List[str]] = ["precision", "recall", "ndcg", "mrr"]
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

    df = pd.merge(y_trues, y_preds, on="userId")
    
    eval_scores = {}
    for metric in metrics:

        if metric == "precision":
            eval_scores[metric] = df[["itemIds_x", "itemIds_y"]].apply(lambda x: precision_at_k(x[0], x[1], k), axis=1).mean()
            eval_scores["map"] = df[["itemIds_x", "itemIds_y"]].apply(lambda x: map_at_k(x[0], x[1], k), axis=1).mean()
        
        if metric == "recall":
            eval_scores[metric] = df[["itemIds_x", "itemIds_y"]].apply(lambda x: recall_at_k(x[0], x[1], k), axis=1).mean()

        if metric == "ndcg":
            eval_scores[metric] = df[["itemIds_x", "itemIds_y"]].apply(lambda x: ndcg_at_k(x[0], x[1], k), axis=1).mean()
        
        if metric == "mrr":
            eval_scores[metric] = df[["itemIds_x", "itemIds_y"]].apply(lambda x: mrr_at_k(x[0], x[1], k), axis=1).mean()
    return eval_scores