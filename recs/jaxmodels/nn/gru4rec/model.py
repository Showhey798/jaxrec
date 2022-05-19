from typing import Optional, Tuple
from flax import struct
import jax
from flax import linen as fnn
from jax import numpy as jnp

from ..config import GRU4RecConfig


class GRU4Rec(fnn.Module):
    config : GRU4RecConfig
    @staticmethod
    def init_hidden( 
        batch_size:int,
        hidden_size:int,
        num_layers:int,
        rng_key:Optional[int]=0
    ):
        return fnn.GRUCell().initialize_carry(
                jax.random.PRNGKey(rng_key), 
                (batch_size,),
                hidden_size*num_layers).reshape((num_layers, -1, hidden_size))
    
    def final_act(self):
        if self.config.final_act == "softmax":
            return fnn.softmax
        elif self.config.final_act == "tanh":
            return fnn.tanh
        elif self.config.final_act == "relu":
            return fnn.relu
        elif self.config.final_act == "log_softmax":
            return fnn.log_softmax
    
    @fnn.compact
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
            x = fnn.Embed(self.config.output_size, self.config.embedding_dim)(inputs)
        else:
            x = jax.nn.one_hot(inputs, self.config.output_size)
        
        for i in range(self.config.num_layers):
            hidden_, x = fnn.GRUCell()(hidden[i], x)
            x = fnn.Dropout(self.config.dropout_hidden)(x, deterministic=deterministic)
            hidden_list += [hidden_]
        
        
        output = fnn.Dense(self.config.output_size)(x)
        output = self.final_act()(output)
        return output, jnp.stack(hidden_list, axis=0)
    