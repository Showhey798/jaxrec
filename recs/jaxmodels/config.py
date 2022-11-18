from typing import Optional, List
from flax import struct

@struct.dataclass
class GRU4RecConfig:
    hidden_size : int
    output_size : int
    num_layers : Optional[int]=1
    embedding_dim : Optional[int]=-1
    batch_size : Optional[int]=64
    dropout_hidden : Optional[float]=0.5
    dropout_input : Optional[int]=0
    final_act : Optional[str]="log_softmax"
    rng_key : Optional[int] = 0
    learning_rate:Optional[float]=0.01
    num_epochs:Optional[int]=500
    early_stop_count:Optional[int]=3
    gamma:Optional[float]=1.

@struct.dataclass
class SASRecConfig:
    hidden_size : int
    num_heads : int
    dropout_rate : float
    num_units:List[int]
    eps:Optional[float]=1e-8
    
    