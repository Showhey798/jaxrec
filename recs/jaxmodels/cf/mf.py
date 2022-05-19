from typing import Dict, Optional
from functools import partial
import numpy as np
from tqdm.notebook import tqdm

import tensorflow as tf
import jax
import jax.numpy as jnp
from jax import random, vmap, jit

from recs.evaluator import evaluate

def init_params( 
    num_users:int, 
    num_items:int, 
    embed_dim:int,
    key:int=0
)->Dict[str, jnp.ndarray]:
    
    rng_key = random.PRNGKey(key)
    rng_key, rng_key_ = random.split(rng_key)
    
    params = {
        "user_embedding": random.truncated_normal(rng_key, lower=0., upper=3., shape=(num_users, embed_dim)),
        "item_embedding": random.truncated_normal(rng_key_, lower=0., upper=3., shape=(num_items, embed_dim))
    }
    
    return params

@jax.jit
def update(
    params:Dict[str, jnp.ndarray],
    data:np.ndarray,
    alpha:float,
    lam:float
)->Dict[str, jnp.ndarray]:
    
    user, item, rating = data[:, 0], data[:, 1], data[:, 2]
    
    u_emb = params.get("user_embedding")[user] # (batch_size, embed_dim)
    i_emb = params.get("item_embedding")[item] # (batch_size, embed_dim)
    
    preds = (u_emb * i_emb).sum(axis=1)
    error = (rating - preds).reshape(len(u_emb), 1)

    params["user_embedding"] = params["user_embedding"].at[user].set(u_emb + alpha * (2 * error * i_emb - lam * u_emb))
    params["item_embedding"] = params["item_embedding"].at[item].set(i_emb + alpha * (2 * error * u_emb - lam * i_emb))
    
    return params, jnp.mean(jnp.square(error))

@partial(jax.jit, static_argnums=(2,))
def predict(
    params:Dict[str, jnp.ndarray],
    u:int,
    k:Optional[int]=10
):
    u_emb = params.get("user_embedding")[u] #(embed_dim,)
    i_emb = params.get("item_embedding")    # (num_items, emebd_dim)
    pred = i_emb@u_emb
    
    pred_ids = jnp.argsort(pred)[::-1][:k]
    
    return pred_ids

def predict_items(params:Dict[str, jnp.ndarray], data:np.ndarray, k:Optional[int]=10):
    users = np.unique(data[:, 0])
    pred_dic = {}
    for u in tqdm(users, desc="predict"):
        pred_dic[u] = np.asarray(mf.predict(params, u, k))
    return pd.DataFrame(pred_dic.items(), columns=["userId", "itemIds"])

def train_one_epoch(
    epoch:int,
    params: Dict[str, jnp.ndarray], 
    data:np.ndarray,
    log_epoch:Optional[int]=1,
    batch_size:Optional[int]=64,
    alpha:Optional[float]=0.01,
    lam:Optional[float]=0.03
):
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    index = jnp.asarray(index)
    num_batches = int(data.shape[0] // batch_size) + 1
    batch_loss = 0.
    for batch in tqdm(range(num_batches), desc="training"):
        batch_idx = index[batch * batch_size: (batch + 1) * batch_size]
        params, loss = update(params, data[batch_idx], alpha, lam)
        batch_loss += loss
    
    batch_loss /= num_batches
    
    return params

@partial(jax.jit, static_argnums=(2,))
def train(
    params:Dict[str, jnp.ndarray], 
    data:np.ndarray, 
    batch_size:Optional[int]=64,
    log_epoch:Optional[int]=1,
    epochs:Optional[int]=10, 
    alpha:Optional[float]=0.01, 
    lam:Optional[float]=0.03
):
    params = jax.lax.fori_loop(0, epochs, lambda epoch_, params_:train_one_epoch(epoch_, params_, data, log_epoch, batch_size, alpha, lam), params)
    return params