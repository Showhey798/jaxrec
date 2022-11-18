from typing import Optional,List
from tqdm.notebook import tqdm
import datetime
import os
import copy
import pandas as pd
import numpy as np

import pickle

import tensorflow as tf
from tensorflow import keras as tfk

import sys
sys.path.append("..")
from evaluator import metrics

class GRU4Rec(tfk.Model):
    
    def __init__(
        self,
        num_items:int,
        seq_len:Optional[int]=3,
        hidden_dim:Optional[int]=100,
        embed_dim:Optional[int]=100,
        dropout_rate:Optional[float]=0.5,
        name="GRU"
    ):
        super(GRU4Rec, self).__init__(name=name)
        
        self._embedding = tfk.layers.Embedding(num_items, embed_dim, mask_zero=True)
        self._gru = tfk.layers.GRU(
            hidden_dim, 
            dropout=dropout_rate)

        self._dense = tfk.layers.Dense(num_items, activation="softmax")
    
    def call(
        self, 
        item_seqs:tf.Tensor,
        training:Optional[bool]=False
    ):
        
        x = self._embedding(item_seqs)
        x = self._gru(x, training=training)
        out = self._dense(x)
        return out


def main():
    
    