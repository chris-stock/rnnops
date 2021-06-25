"""
Methods for training RNNs via backprop/autodiff methods. Todo.
"""
import tensorflow as tf

def clip_gradients(clip_at):

    if clip_at is None:
        return None
    else:
        return {
        'method': tf.clip_by_global_norm,
        'args': (10.0,)  # (10.0,)
    }