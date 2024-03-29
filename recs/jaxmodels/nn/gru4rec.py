from typing import Optional, Tuple
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import pandas as pd

import jax
from jax import numpy as jnp

import optax
from flax import struct
from flax import linen as nn
from flax.training import train_state
from flax.training import common_utils

import tensorflow as tf

from ..config import GRU4RecConfig

def bpr_loss(
    y_true:jnp.ndarray,
    y_pred:jnp.ndarray
):
    true_logits = jnp.diag(y_pred[:, y_true]) # (batch_size, )
    true_logits = jnp.expand_dims(true_logits, axis=-1) # (batch_size, 1)
    diff = true_logits - y_pred
    loss = -jnp.mean(jnp.log(nn.sigmoid(diff)))
    return loss
    

def top1_loss(
    y_true:jnp.ndarray, # (batch_size, )
    y_pred:np.ndarray   # (batch_size, num_items)
):
    true_logits = jnp.diag(y_pred[:, y_true]) # (batch_size, )
    true_logits = jnp.expand_dims(true_logits, axis=-1) # (batch_size, 1)
    diff = y_pred - true_logits
    
    loss = jnp.log(nn.sigmoid(diff)).mean() + nn.sigmoid(y_pred**2).mean() # (batch_size, )
    return loss

def cross_entropy_loss(
    y_true:jnp.ndarray, # (batch_size, )
    y_pred:np.ndarray   # (batch_size, num_items)
):
    
    logits = y_pred[:, y_true]

    one_hot_labels = jax.nn.one_hot(
        jnp.arange(y_true.shape[0]), 
        num_classes=y_true.shape[0])

    loss = -jnp.mean(jnp.sum(one_hot_labels * logits, axis=1))
    return loss


class LossFnuc():
    def __init__(self, lossname:Optional[str]="top1"):
        if lossname == "top1":
            self._loss = top1_loss
        elif lossname == "cross-entorpy":
            self._loss = cross_entropy_loss
    
    def __call__(self, y_true, y_pred):
        return self._loss(y_true, y_pred)
    

class GRU4Rec(nn.Module):
    config : GRU4RecConfig
    @staticmethod
    def init_hidden( 
        batch_size:int,
        hidden_size:int,
        num_layers:int,
        rng_key:Optional[int]=0
    ):
        return nn.GRUCell().initialize_carry(
                jax.random.PRNGKey(rng_key), 
                (batch_size,),
                hidden_size*num_layers).reshape((num_layers, -1, hidden_size))
    
    def final_act(self):
        if self.config.final_act == "softmax":
            return nn.softmax
        elif self.config.final_act == "tanh":
            return nn.tanh
        elif self.config.final_act == "relu":
            return nn.relu
        elif self.config.final_act == "log_softmax":
            return nn.log_softmax
    
    @nn.compact
    def __call__(
        self,
        inputs:jnp.ndarray,
        hidden:jnp.ndarray,
        deterministic:Optional[bool]=True
    ):
        """
        Args :
            inputs (batch_size,) : 
                    batch of item indices from a session-paralell mini-batch
            hidden (num_layers, batch_size, hidden_size):
            deterministic:

        """
        hidden_list = []
        if self.config.embedding_dim > 0:
            x = nn.Embed(self.config.output_size, self.config.embedding_dim)(inputs)
        else:
            x = jax.nn.one_hot(inputs, self.config.output_size)
        
        for i in range(self.config.num_layers):
            hidden_, x = nn.GRUCell()(hidden[i], x)
            x = nn.Dropout(self.config.dropout_hidden)(x, deterministic=deterministic)
            hidden_list += [hidden_]
        
        
        output = nn.Dense(self.config.output_size)(x)
        output = self.final_act()(output)
        return output, jnp.stack(hidden_list, axis=0)

    
def create_train_state(key, config):
    model = GRU4Rec(config)
    hidden = model.init_hidden(config.batch_size, config.hidden_size, config.num_layers)
    params = model.init(key, jnp.ones((config.batch_size,), dtype=jnp.int32), hidden)["params"]
    tx = optax.adam(config.learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx), hidden, model




def train(
    config:GRU4RecConfig,
    model:nn.Module,
    state:train_state.TrainState,
    train_data:tf.data.Dataset,
    train_len:int,
    hidden:Optional[jnp.ndarray]=None
):
    if hidden is None:
        hidden = model.init_hidden(config.batch_size, config.hidden_size, config.num_layers)
        
    best_loss = np.inf
    stop_count = 0
    losses = []
    
    @jax.jit
    def train_step(state, x, y, mask, hidden):
        
        if len(mask) != 0:
            hidden = hidden.at[:, mask, :].set(0)

        def loss_fn(params, hidden):
            logits, hidden = model.apply({"params":params}, x, hidden) # (batch_size, num_items)
            logits, targets = jnp.squeeze(logits), jnp.squeeze(y)

            # loss = top1_loss(targets, logits)
            # loss = bpr_loss(targets, logits)
            
            loss = cross_entropy_loss(targets, logits)
            return loss, hidden
        
        (loss, hidden), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, hidden)
        state = state.apply_gradients(grads=grads)
        return state, loss, hidden
    
    for epoch in range(config.num_epochs):
        batch_loss = 0.
        with tqdm(
            train_data.as_numpy_iterator(), desc="[Epoch %d] training"%(epoch+1), total=train_len, ncols=100) as ts:
            for i, batch in enumerate(ts):

                batch = common_utils.shard(batch)
                inputs, targets, masks = batch[0], batch[2], batch[1]
                    
                state, loss, hidden = train_step(state, inputs, targets, masks, hidden)
                batch_loss += np.asarray(loss)

                ts.set_postfix_str("loss=%4f"%(batch_loss / (i+1)))
        
        batch_loss = batch_loss / (i + 1)
        losses += [batch_loss]
        
        if best_loss > batch_loss:
            best_loss = batch_loss
            stop_count = 0
        else:
            stop_count += 1

        if stop_count > config.early_stop_count:
            break
        
    
    return state, hidden, losses

def predict(
    config:GRU4RecConfig,
    model:nn.Module,
    state:train_state.TrainState,
    test_data:tf.data.Dataset,
    test_len,
    k:Optional[int]=100,
    sessionkey:Optional[str]="sessionId"
):

    hidden = model.init_hidden(config.batch_size, config.hidden_size, config.num_layers)
    pred_df = pd.DataFrame(columns=[sessionkey, "predIds", "trueIds"])
    for i, batch in tqdm(enumerate(test_data.as_numpy_iterator()), desc="predicting", total=test_len, ncols=100):
        batch = common_utils.shard(batch)
        inputs, targets, masks, sessId = jnp.squeeze(batch[0]), jnp.squeeze(batch[2]), jnp.squeeze(batch[1]), jnp.squeeze(batch[3])
        masks = jnp.expand_dims(masks, axis=-1)
        hidden = masks * hidden
        logits, hidden = jax.jit(model.apply)({"params":state.params}, inputs, hidden)

        _, indices = jax.lax.top_k(logits, k=k)

        pred_df = pd.concat([
            pred_df, 
            pd.DataFrame([np.asarray(sessId), np.asarray(indices), np.asarray(targets)]).T.rename({0:sessionkey, 1:"predIds", 2:"trueIds"},axis=1)
        ], axis=0)
    
    return pred_df