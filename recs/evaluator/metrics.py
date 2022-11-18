import numpy as np

def hit_at_k(y_true, y_pred, k=10):
    if type(y_true) is not np.ndarray:
        y_true = np.asarray([y_true])
    return len(set(y_true) & set(y_pred[:k]))

def precision_at_k(
    y_trues: np.ndarray, 
    y_preds: np.ndarray,
    k : int=10
):
    if type(y_trues) != np.ndarray:
        y_trues=np.asarray([y_trues])
        
    # Precision@k = (y_trues & y_preds[:k]) / k
    return len(set(y_trues) & set(y_preds[:k])) / k

def map_at_k(
    y_trues: np.ndarray, 
    y_preds: np.ndarray,
    k : int=10
):
    map_at_k_list = [precision_at_k(y_trues, y_preds, k_+1) for k_ in range(k)]
    return np.mean(map_at_k_list)


def recall_at_k(
    y_trues: np.ndarray, 
    y_preds: np.ndarray, 
    k : int=10
):
    if type(y_trues) != np.ndarray:
        y_trues=np.asarray([y_trues])
    # Recall@k = len(y_trues & y_preds) / len(y_trues)
    return len(set(y_trues) & set(y_preds[:k])) / len(set(y_trues))

def ndcg_at_k(
    y_trues: np.ndarray, 
    y_preds: np.ndarray,
    k: int=10
):
    if type(y_trues) != np.ndarray:
        y_trues=np.asarray([y_trues])
    dcg_score = 1. / np.log((np.where(y_trues[:k] == y_preds[:k])[0] + 2))
    if len(dcg_score) == 0:
        return 0.
    else:
        return dcg_score[0]
    
     


def mrr_at_k(
    y_trues: np.ndarray,
    y_preds: np.ndarray,
    k: int=10
):  
    if type(y_trues) != np.ndarray:
        y_trues=np.asarray([y_trues])
    score = 0
    rank = np.where(y_trues[:k] == y_preds[:k])[0]
    if len(rank) != 0:
        score = np.sum(1. / (rank + 1), axis=-1)
    return score