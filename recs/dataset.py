
from typing import Optional
import pandas as pd
import numpy as np
import tensorflow as tf


def session_parallel_dataset(
    path:Optional[str]="/home/inoue/work/dataset/RC15/derived/train.df",
    sessionkey:Optional[str]="sessionId",
    itemkey:Optional[str]="itemId",
    timekey:Optional[str]="timestamp",
    batch_size:Optional[int]=32,
    prefech_size:Optional[int]=tf.data.experimental.AUTOTUNE
)->tf.data.Dataset:
    df = pd.read_pickle(path)
    
    df.sort_values([timekey, sessionkey], inplace=True)
    click_offsets = np.zeros(df[sessionkey].nunique()+1, dtype=np.int32)
    click_offsets[1:] = df.groupby(sessionkey).size().cumsum()
    session_start = df.groupby(sessionkey)[timekey].min().values
    session_idx_arr = np.argsort(session_start)
    total_length = int(len(df) // batch_size) + 1
    
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
                masks = np.ones(batch_size)
                masks[mask] = 0
                yield {"input":idx_input, "target":idx_target, "mask":masks}

            start += (minlen-1)
            mask = np.arange(batch_size)[(end- start) <= 1]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]
    
    datasets = tf.data.Dataset.from_generator(
        generator, output_types={"input":tf.int32, "target":tf.int32, "mask":tf.float32}, output_shapes={"input":(batch_size,), "target":(batch_size,), "mask":(batch_size,)}
    )
    datasets = datasets.prefetch(prefech_size)
    return datasets, total_length