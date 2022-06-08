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
    
    for user, pos, neg in data:
    
        u_emb = params.get("user_embedding")[user] # (batch_size, embed_dim)
        pos_emb = params.get("item_embedding")[pos] # (batch_size, embed_dim)
        neg_emb = params.get("item_embedding")[neg]

        y_pos = (u_emb * pos_emb).sum(axis=1)
        y_neg = (u_emb * neg_emb).sum(axis=1)
        exp_x = jnp.exp(-(y_pos - y_neg))
        mult = -exp_x / (1.0 + exp_x)
        mult = mult.reshape(-1, 1)
        grad_user = pos_emb - neg_emb

        params["user_embedding"] = params["user_embedding"].at[user].set(u_emb + alpha * (mult * grad_user - lam * u_emb))
        params["item_embedding"] = params["item_embedding"].at[pos].set(pos_emb + alpha * (mult * u_emb - lam * pos_emb))
        params["item_embedding"] = params["item_embedding"].at[neg].set(neg_emb + alpha * (-mult * u_emb - lam * neg_emb))
    
    error = jnp.log(1./(1. + exp_x))

    return params, jnp.sum(error)

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