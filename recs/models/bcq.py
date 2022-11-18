from typing import Optional,List, Dict
from tqdm.notebook import tqdm
import datetime
import os
import copy
import sys
sys.path.append("..")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from recs.evaluator import metrics

import tensorflow as tf
from tensorflow import keras as tfk
from tensorboard.plugins.hparams import api as hp


class BCQNet(tfk.layers.Layer):
    def __init__(
        self,
        num_items:int,
        seq_len:Optional[int]=3,
        hidden_dim:Optional[int]=100,
        embed_dim:Optional[int]=100,
        dropout_rate:Optional[int]=0.5,
        name="QNet"
    ):
        super(QNet, self).__init__(name=name)
        
        self._embedding = tfk.layers.Embedding(num_items, embed_dim, mask_zero=True)
        self._gru = tfk.layers.GRU(
            hidden_dim, 
            dropout=dropout_rate)

        self._qvalue_dense = tfk.layers.Dense(num_items+1, activation=None)
        self._imt_dense = tfk.layers.Dense(num_items, activation="softmax")
        self._lambda = tfk.layers.Lambda(
            lambda x: tf.expand_dims(x[:, 0], axis=-1) + x[:, 1:] - tf.reduce_mean(x[:, 1:], axis=-1, keepdims=True),
            output_shape=(num_items, )
        )
    
    def call(
        self, 
        item_seqs:tf.Tensor, # (batch_size, seq_len)
        training:Optional[bool]=False,
    ):
        x = self._embedding(item_seqs)
        x = self._gru(x, training=training)
        
        i = self._imt_dense(x)
        x = self._qvalue_dense(x)
        qvalue = self._lambda(x)
        return qvalue, i
    
class BCQRec(tfk.Model):
    
    def __init__(
        self,
        num_items:int,
        seq_len:int,
        hparams:Dict[str, hp.Hparam]
        tau:Optional[int]=1.,
        update_count:Optional[int]=20,
        k:Optional[int]=20,
        name="BCQRec"
    ):
        super(BCQRec, self).__init__(name=name)
        self._qmodel = BCQNet(num_items, seq_len, hparams["hidden_dim"], hparams["embed_dim"], hparams["dropout_rate"], name="BCQ")
        self._target_qmodel = copy.deepcopy(self._qmodel)
        
        self._num_items = num_items
        
        self._topk = k
        self._threshold = hparams["threshold"]
        self._gamma = hparams["gamma"]
        self._lam = hparams["lam"]
        self._iterations = 0
        self._update_count = update_count
        self._loss_tracker = tfk.metrics.Mean(name="loss")
        self._recall_tracker = tfk.metrics.Recall(name="recall")
        
        dummy_state = tf.zeros((1, seq_len), dtype=tf.int32)
        self._qmodel(dummy_state)
        self._target_qmodel(dummy_state)
        
    
    def compile(
        self, 
        optimizer:tfk.optimizers.Optimizer
    ):
        super(BCQRec, self).compile()
        self.q_loss = tfk.losses.Huber()
        self.imt_loss = tfk.losses.CategoricalCrossentropy()
        self.optimizer = optimizer
    
    def call(
        self, 
        state:tf.Tensor
    ):
        q, imt =  self._qmodel(state)
        q = imt*q + (1-imt)*-1e8
        return q
    
    def train_step(self, data):
        self._iterations += 1
        state, action, reward, n_state, done = data
        onehot_act = tf.one_hot(action-1, depth=self._num_items)
        
        with tf.GradientTape() as tape:
            # Compute Target Q-value
            q, imt = self._qmodel(n_state)
            imt = tf.cast(imt / tf.reduce_max(imt, axis=1, keepdims=True) > self._threshold, dtype=tf.float32)

            next_action_val = imt*q + (1-imt)*-1e8
            next_action = tf.argmax(next_action_val, axis=-1)
            onehot_next_act = tf.one_hot(next_action, depth=self._num_items)
            
            q, imt = self._target_qmodel(n_state)
            target_q = reward + (1-done)*self._gamma*tf.reduce_sum(q*onehot_next_act, axis=-1)
            target_q = tf.stop_gradient(target_q)
            
            # Get Current Qvalue estimate
            
            current_q, imt = self._qmodel(state)
            current_q = tf.reduce_sum(current_q*onehot_act, axis=-1)
            
            q_loss = self.q_loss(target_q, current_q)
            imt_loss = self.imt_loss(onehot_act, imt)
            
            loss = q_loss + imt_loss + self._lam * tf.reduce_mean(tf.math.pow(imt, 2))
        
        grads = tape.gradient(loss, self._qmodel.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self._qmodel.trainable_variables))
        
        self._loss_tracker.update_state(loss)
        
        if self._iterations % self._update_count == 0:
            for param, tarparam in zip(self._qmodel.trainable_variables, self._target_qmodel.trainable_variables):
                tar_param.assign(self._tau*param + (1-self._tau)*tarparam)
        
        return {"loss":self._loss_tracker.result()}
    
    def test_step(self, data):
        state, target, _, _, _ = data
        target = tf.one_hot(target-1, depth=self._num_items)
        target = tf.cast(target, dtype=tf.int32)

        qvalue = self(state)
        topkitem = tf.math.top_k(qvalue, k=self._topk)[1]
        topkitem = tf.reduce_sum(tf.one_hot(topkitem, depth=self._num_items), axis=1)
        topkitem = tf.cast(topkitem, dtype=tf.int32)
        
        self._recall_tracker.update_state(target, topkitem)
        
        return {"recall":self._recall_tracker.result()}
    
    @property
    def metrics(self):
        return [self._loss_tracker, self._recall_tracker]

    