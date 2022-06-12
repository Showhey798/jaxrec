
from typing import Optional
import pandas as pd
import numpy as np
import tensorflow as tf


def session_parallel_dataset(
    path:Optional[str]="~/work/dataset/RC15/derived/train.df",
    sessionkey:Optional[str]="sessionId",
    itemkey:Optional[str]="itemId",
    timekey:Optional[str]="timestamp",
    issort:Optional[bool]=True,
    batch_size:Optional[int]=32,
    prefetch_size:Optional[int]=tf.data.experimental.AUTOTUNE
)->tf.data.Dataset:
    df = pd.read_pickle(path)
    
    if issort:
        df.sort_values([timekey, sessionkey], inplace=True)
    click_offsets = np.zeros(df[sessionkey].nunique()+1, dtype=np.int32)
    click_offsets[1:] = df.groupby(sessionkey).size().cumsum()
    session_start = df.groupby(sessionkey)[timekey].min().values
    session_idx_arr = np.argsort(session_start)
    total_length = int((len(df) - len(click_offsets)) // batch_size)
    
    def generator():
        iters = np.arange(batch_size)
        maxiter=iters.max()
        mask = []
        finished=False
        start, end = click_offsets[session_idx_arr[iters]], click_offsets[session_idx_arr[iters] + 1]
        while not finished:
            minlen = (end - start).min()
            idx_target = df[itemkey].values[start]

            for i in range(minlen - 1):
                idx_input = idx_target
                idx_target = df[itemkey].values[start + i + 1]
                sessId = df[sessionkey].values[start + i + 1]
                masks = np.ones(batch_size)
                masks[mask] = 0
                yield {"input":idx_input, "target":idx_target, "mask":masks, "sessId":sessId}

            start += (minlen-1)
            mask = np.arange(batch_size)[(end-start) <= 1]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]
    
    with tf.device("/device:CPU:0"):
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types = {"input" : tf.int32, "target" : tf.int32, "mask":tf.float32, "sessId":tf.int32},
            output_shapes = {"input":(batch_size,), "target":(batch_size, ), "mask":(batch_size, ), "sessId":(batch_size, )}
        )

    return dataset, total_length

def negative_sample_dataset(
    path:Optional[str]="~/work/dataset/RC15/derived/train.df",
    sessionkey:Optional[str]="sessionId",
    itemkey:Optional[str]="itemId",
    timekey:Optional[str]="timestamp",
    batch_size:Optional[int]=32,
    prefetch_size:Optional[int]=tf.data.experimental.AUTOTUNE
):
    try:
        df = pd.read_pickle(path)
    except:
        df = pd.read_csv(path)
    df.sort_values([timekey, sessionkey], inplace=True)
    
    sess_item_dict = df.groupby(sessionkey)[itemkey].unique().reset_index()
    sess_item_dict = dict(zip(sess_item_dict[sessionkey], sess_item_dict[itemkey]))
    max_item = df[itemkey].max() + 1
    max_user = df[sessionkey].max() + 1
    sessions = df[sessionkey].unique()
    
    def generator():    
        for s in sessions:
            items = sess_item_dict[s]
            for i in items:
                neg = np.random.randint(max_item)
                while neg in items:neg = np.random.randint(max_item)
                yield (s, i, neg)
    
    with tf.device("/device:CPU:0"):
        train_data = tf.data.Dataset.from_generator(
            generator,
            output_shapes= (3,),
            output_types = tf.int32,
        )
        train_data = train_data.batch(batch_size).prefetch(prefetch_size)
        
    num_batch = int(len(df)//batch_size) + 1        
    return train_data, num_batch, max_item, max_user
    