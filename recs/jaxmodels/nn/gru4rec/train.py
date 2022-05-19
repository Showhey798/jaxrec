from functools import partial
import os
import time

from flax import linen as nn
from flax.training import train_state
import optax

import jax 
from jax import numpy as jnp
from jax import random
import numpy as np
import tensorflow as tf

def create_train_state(key, config, model):
    hidden = model.init_hidden(config.batch_size, config.hidden_size, config.num_layers)
    params = model.init(key, jnp.ones((config.batch_size,), dtype=jnp.float32), hidden)["params"]
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, x, y, mask, hidden, model):
    mask = jnp.expand_dims(mask, axis=-1)
    hidden = mask * hidden
    def loss_fn(params, hidden):
        logits, hidden = model.apply({"params":params}, x, hidden)
        one_hot_labels = jax.nn.one_hot(y, num_classes=logits.shape[-1])
        logits, one_hot_labels = jnp.squeeze(logits), jnp.squeeze(one_hot_labels)
        loss = -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))
        return loss, hidden
    (loss, hidden), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, hidden)
    state = state.apply_gradients(grads=grads)
    metrics = {
        "loss":loss
    }
    return state, metrics, hidden