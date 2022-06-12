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

from .config import GRU4RecConfig


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
    params = model.init(key, jnp.ones((config.batch_size,), dtype=jnp.float32), hidden)["params"]
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
        
    before_loss = np.inf
    stop_count = 0
    losses = []
    
    @jax.jit
    def train_step(state, x, y, mask, hidden):
        mask = jnp.expand_dims(mask, axis=-1)
        hidden = mask * hidden
        def loss_fn(params, hidden):
            logits, hidden = model.apply({"params":params}, x, hidden) # (batch_size, num_items)
            logits, targets = jnp.squeeze(logits), jnp.squeeze(y)
            logits = logits[:, targets]

            one_hot_labels = jax.nn.one_hot(
                jnp.arange(targets.shape[0]), 
                num_classes=targets.shape[0]
            )

            #one_hot_labels = jax.nn.one_hot(targets, num_classes=logits.shape[1])
            loss = -jnp.mean(jnp.sum(one_hot_labels * logits, axis=1))
            return loss, hidden
        (loss, hidden), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, hidden)
        state = state.apply_gradients(grads=grads)
        return state, loss, hidden
    
    for epoch in range(config.num_epochs):
        batch_loss = 0.
        with tqdm(
            train_data.as_numpy_iterator(), desc="[Epoch %d] training"%epoch, total=train_len) as ts:
            for i, batch in enumerate(ts):
                batch = common_utils.shard(batch)
                inputs, targets, masks, sess = batch["input"], batch["target"], batch["mask"], batch["sessId"]
                state, loss, hidden = train_step(state, inputs, targets, masks, hidden)
                batch_loss += np.asarray(loss)

                ts.set_postfix_str("loss=%4f"%(batch_loss / (i+1)))
        
        batch_loss = batch_loss / (i + 1)
        losses += [batch_loss]
        
        if before_loss > batch_loss:
            before_loss = batch_loss
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
    for i, batch in tqdm(enumerate(test_data.as_numpy_iterator()), desc="predicting", total=test_len):
        batch = common_utils.shard(batch)
        inputs, targets, masks, sessId = jnp.squeeze(batch["input"]), jnp.squeeze(batch["target"]), jnp.squeeze(batch["mask"]), jnp.squeeze(batch["sessId"])
        masks = jnp.expand_dims(masks, axis=-1)
        hidden = masks * hidden
        logits, hidden = jax.jit(model.apply)({"params":state.params}, inputs, hidden)

        _, indices = jax.lax.top_k(logits, k=k)

        pred_df = pd.concat([
            pred_df, 
            pd.DataFrame([np.asarray(sessId), np.asarray(indices), np.asarray(targets)]).T.rename({0:sessionkey, 1:"predIds", 2:"trueIds"},axis=1)
        ], axis=0)
    
    return pred_df