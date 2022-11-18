from typing import Optional
import jax
from jax import numpy as jnp

from flax import linen as nn

class GRU(nn.Module):

    @staticmethod
    def init_hidden(batch_size:int, hidden_size:int):
        return nn.GRUCell.initialize_carry(jax.random.PRNGKey(0), (batch_size, ), hidden_size)
    
    @nn.compact
    def __call__(
        self, 
        inputs, # (batch_size, seq_len, embed_dim)
        hidden # (batch_size, hidden_size)
    ):
        
        for i in range(inputs.shape[1]):
            _, hidden =  nn.GRUCell()(hidden, inputs[:, i, :])
        return hidden

class QNet(nn.Module):
    embed_dim: int
    output_size: int
    hidden_dim: int
    dropout_rate: float
    

    @nn.compact
    def __call__(
        self,
        inputs:jnp.ndarray,     # (batch_size, seq_len)
        training: Optional[bool]=True
    ):
        
        hidden = GRU().init_hidden(inputs.shape[0], self.hidden_dim)
        
        x = nn.Embed(self.embed_dim, self.output_size+1)(inputs)
        x = GRU()(x, hidden)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
    
        behavior = nn.Dense(self.output_size)(x)
        behavior = nn.softmax(behavior)

        qvalue = nn.Dense(self.output_size)(x)
        return behavior, qvalue