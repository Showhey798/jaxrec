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

class SQNGRUModel(tfk.layers.Layer):
    
    def __init__(
        self,
        num_items:int,
        seq_len:Optional[int]=3,
        hidden_dim:Optional[int]=100,
        embed_dim:Optional[int]=100,
        dropout_rate:Optional[float]=0.5,
        name="GRU"
    ):
        super(SQNGRUModel, self).__init__(name=name)
        
        
        self._embedding = tfk.layers.Embedding(num_items, embed_dim, mask_zero=True)
        self._gru = tfk.layers.GRU(
            hidden_dim, 
            dropout=dropout_rate)

        self._dense = tfk.layers.Dense(num_items, activation="softmax")
        self._qvalue_dense = tfk.layers.Dense(num_items, activation=None)
    
    def call(
        self, 
        item_seqs:tf.Tensor, # (batch_size, seq_len)
        training:Optional[bool]=False,
        is_next:Optional[bool]=False
    ):
        x = self._embedding(item_seqs)
        x = self._gru(x, training=training)
        qvalue = self._qvalue_dense(x)
        if is_next:
            return qvalue
        out = self._dense(x)
            
        return out, qvalue

class SQNGRU4Rec(tfk.Model):
    
    def __init__(
        self,
        num_items:int,
        seq_len:Optional[int]=3,
        hidden_dim:Optional[int]=100,
        embed_dim:Optional[int]=100,
        dropout_rate:Optional[float]=0.5,
        gamma:Optional[float]=1.,
        name="SQN-GRUModel"
    ):
        super(SQNGRU4Rec, self).__init__(name=name)
        self._gamma = gamma
        self._num_items = num_items
        
        self._gmodel = SQNGRUModel(num_items, seq_len, hidden_dim, embed_dim, dropout_rate, name="SQNGRU")
        self._target_gmodel = SQNGRUModel(num_items, seq_len, hidden_dim, embed_dim, dropout_rate, name="TargetSQNGRU")
        
        self._loss_tracker = tfk.metrics.Mean(name="loss")
        self._tdloss_tracker = tfk.metrics.Mean(name="TD Error")
            
        dummy_state = tf.zeros((1, seq_len), dtype=tf.int32)
        self._gmodel(dummy_state)
        self._target_gmodel(dummy_state)

    
    def compile(self, g_loss, q_loss, optimizer):
        super(SQNGRU4Rec, self).compile()
        self.g_loss = g_loss
        self.q_loss = q_loss
        self.optimizer = optimizer
    
    def call(self, states):
        x, _ = self._gmodel(states)
        return x
    
    def __train_step(self, data):
        state, action, reward, n_state, done = data
        onehot_act = tf.one_hot(action-1, depth=self._num_items)

        
        with tf.GradientTape() as tape:
            out, qvalue = self._gmodel(state, training=True)
            n_qvalue = self._gmodel(n_state, training=True, is_next=True)
            n_qvalue_ = self._target_gmodel(n_state, training=True, is_next=True)
            
            greedy_a = tf.argmax(n_qvalue, axis=-1)
            onehot_greedy_a = tf.one_hot(greedy_a, depth=self._num_items)
            
            Lq = reward + (1.0 - done) * self._gamma * tf.reduce_sum(n_qvalue_*onehot_greedy_a, axis=-1)
            Lq = tf.stop_gradient(Lq)
            Lq = self.q_loss(Lq, tf.reduce_sum(qvalue*onehot_act,axis=-1))
            
            Ls = self.g_loss(onehot_act, out)
            loss = Lq + Ls
            
            
        grads = tape.gradient(loss, self._gmodel.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self._gmodel.trainable_variables))
        self._loss_tracker.update_state(loss)

        self._tdloss_tracker.update_state(Lq)
        
        return {"loss": self._loss_tracker.result(), "TD Error":self._tdloss_tracker.result()}
    
    def __tar_train_step(self, data):
        state, action, reward, n_state, done = data
        onehot_act = tf.one_hot(action-1, depth=self._num_items)

        
        with tf.GradientTape() as tape:
            out, qvalue = self._target_gmodel(state, training=True)
            n_qvalue = self._target_gmodel(n_state, training=True, is_next=True)
            n_qvalue_ = self._gmodel(n_state, training=True, is_next=True)
            
            greedy_a = tf.argmax(n_qvalue, axis=-1)
            onehot_greedy_a = tf.one_hot(greedy_a, depth=self._num_items)
            
            Lq = reward + (1.0 - done) * self._gamma * tf.reduce_sum(n_qvalue_*onehot_greedy_a, axis=-1)
            Lq = tf.stop_gradient(Lq)
            Lq = self.q_loss(Lq, tf.reduce_sum(qvalue*onehot_act,axis=-1))
            
            Ls = self.g_loss(onehot_act, out)
            loss = Lq + Ls
            
            
        grads = tape.gradient(loss, self._target_gmodel.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self._target_gmodel.trainable_variables))
        self._loss_tracker.update_state(loss)

        self._tdloss_tracker.update_state(Lq)
        
        return {"loss": self._loss_tracker.result(), "TD Error":self._tdloss_tracker.result()}
        
    def train_step(self, data):
        
        if np.random.uniform(0, 1) <= 0.5:
            loss_hist = self.__train_step(data)
        else:
            loss_hist = self.__tar_train_step(data)
        
        return loss_hist

    @property
    def metrics(self):
        return [self._loss_tracker]
    
def main():
    dataname="diginetica"
    default_logdir = "/home/inoue/work/recs/"
    log_dir =  os.path.join(default_logdir, "logs/%s/"%dataname+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    train = pickle.load(open(
        "/home/inoue/work/dataset/%s/derived/mdp_train.df"%dataname, "rb"
    ))
    data = pd.read_pickle("~/work/dataset/%s/derived/train.df"%dataname)
    testdata = pd.read_pickle("~/work/dataset/%s/derived/test.df"%dataname)

    # ハイパーパラメータ設定
    num_items = max(data.itemId.max(), testdata.itemId.max())
    emb_dim = 64
    hidden_dim = 64
    seq_len = train[1].shape[1]
    batch_size=500
    num_epochs = 10
    k = 20

    train_data = tf.data.Dataset.from_tensor_slices(
        (train[1],#.astype(np.int32),
         train[2],#.astype(np.int32), 
         train[3],#.astype(np.float32), 
         train[4],#.astype(np.int32), 
         train[5].astype(np.float32))).shuffle(len(train[0])).batch(batch_size)
    
    # モデルの作成とコンパイル
    model = SQNGRU4Rec(num_items, seq_len, hidden_dim, emb_dim, dropout_rate=0.1, gamma=0.5)
    model.compile(
        q_loss=tfk.losses.Huber(), 
        g_loss=tfk.losses.CategoricalCrossentropy(),
        optimizer=tfk.optimizers.Adam(learning_rate=0.01)
    )
    model.build(input_shape=(1,seq_len))
    
    hist = model.fit(
        train_data, 
        epochs=num_epochs, 
        callbacks=[
            tfk.callbacks.TensorBoard(log_dir=log_dir), 
            tfk.callbacks.ModelCheckpoint(
                filepath=os.path.join(default_logdir, "params/SQNGRU/checkpoint"),
                save_weights_only=True,
                monitor="loss",
                mode="min",
                save_best_only=True)]
    )
    
    
    # 学習結果の評価
    test = pickle.load(open("/home/inoue/work/dataset/%s/derived/mdp_test.df"%dataname, "rb"))
    test_data = tf.data.Dataset.from_tensor_slices(
        (test[0],test[1],test[2])).shuffle(len(test[0])).batch(batch_size)
    
    df = pd.DataFrame(columns=["sessionId", "recIds", "choiceId"])
    for batch in tqdm(test_data):
        sess, state, target = batch
        pred_score = model(state)
        topkitem = tf.math.top_k(pred_score, k=k)[1].numpy() + 1
        tmp = pd.DataFrame(
            [sess.numpy(), topkitem, target.numpy()]).T
        tmp.columns = ["sessionId", "recIds", "choiceId"]
        df = pd.concat([df, tmp], axis=0)

    
    ndcg_at_k = df[["recIds", "choiceId"]].apply(lambda x: metrics.ndcg_at_k(x[1], x[0], k=k), axis=1).mean()
    hit_at_k = df[["recIds", "choiceId"]].apply(lambda x: metricshit_at_k(x[1], x[0], k=k), axis=1).mean()
    
    return hist, df, ndcg_at_k, hit_at_k

if __name__ == "__main__":
    hist, eval_df, ndcg, hit = main()