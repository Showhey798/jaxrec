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

from ..config import SASRecConfig

class FeedForforward(nn.Module):
    config:SASRecConfig
    @nn.compact
    def __call__(self, x):
        outputs = nn.Dense(self.config.num_units[0])(x)
        outputs = nn.relu()(outputs)
        outputs = nn.Dropout(self.config.dropout_rate)(outputs)
        outputs = nn.Dense(self.config.num_units[1])(outputs)
        outputs = nn.Dropout(self.config.dropout_rate)(outputs)
        outputs += x
        return outputs

class AttentionBlock(nn.Module):
    config:SASRecConfig
    @nn.compact
    def __call__(self, x, mask, deterministic=False):
        
        norm_x = nn.LayerNorm(epsilon=self.config.eps)(x)
        outputs = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            dropout_rate=self.config.dropout_rate
        )(norm_x, x, mask, deterministic)
        outputs = nn.LayerNorm(epsilon=self.config.eps)(outputs)
        outputs = nn.FeedForforward(self.config)(outputs)
        return outputs * mask
        

class SASRec(nn.Module):
    config:SASRecConfig
    
    @nn.compact
    def __call__(
        self, 
        input_seq,
        pos,
        neg=None,
        training=False
    ):
        mask = jnp.expand_dims(input_seq != 0, -1)
        seq = nn.Embedding(self.config.outputsize+1)