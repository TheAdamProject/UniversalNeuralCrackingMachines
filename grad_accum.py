import tensorflow as tf
import utils

def gradient_scaling(Ga, scale):
    n = len(Ga)
    Gb = [None] * n
    for i in range(n):
        a = Ga[i]
        # handle tf gradient for embedding
        if type(a) is tf.IndexedSlices:
            a = tf.convert_to_tensor(a)
        Gb[i] = a / scale
    return Gb

def gradient_scaled_sum(Ga, Gb, scale):
    assert len(Ga) == len(Gb)
    n = len(Ga)
    Gc = [None] * n
    for i in range(n):
        b = Gb[i]
        # handle tf gradient for embedding
        if type(b) is tf.IndexedSlices:
            b = tf.convert_to_tensor(b)
        Gc[i] = Ga[i] + (b / scale) 
    return Gc

