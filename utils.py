import tensorflow as tf
import numpy as np
import string
import os

META = ['<PAD>', '<START>', '<END>']
UNK = ['<UNK>']

def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        ...

def make_vocab_passwords(num_oov_buckets=1, is_username=False):
    if is_username:
        C = list(string.printable[:36])
    else:
        C = list(string.printable[:-5])

    D = META + C

    vocab_size = len(D) + num_oov_buckets
    init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(D),
        values=tf.constant(np.arange(len(D)), dtype=tf.int64)
    )

    char_id_table = tf.lookup.StaticVocabularyTable(
        init,
        num_oov_buckets=num_oov_buckets
    )
    
    D += [UNK]
    return char_id_table, D

def make_vocab(path, num_oov_buckets=1):
    def read_vocab(path):
        with open(path, 'r') as f:
            vocab = [l[:-1] for l in f]
        return vocab

    D = read_vocab(path)

    vocab_size = len(D) + num_oov_buckets
    init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(D),
        values=tf.constant(np.arange(len(D)), dtype=tf.int64)
    )

    id_table = tf.lookup.StaticVocabularyTable(
        init,
        num_oov_buckets=num_oov_buckets
    )
    
    D += [UNK]
    return id_table, D


def print_model_size(model):
    model_mem_footprint = (model.count_params() * 4) // (10 ** 6)
    print("MODEL_MEM: %dMB" % model_mem_footprint)
    
    
def makeNameTest(name, score):
    if name:
        return '%s_%s' % (score, name)
    else:
        return score

def flush_metric(iteration, metric, tfb_log=False):
    value = metric.result()
    if tfb_log:
        name = metric.name
        tf.summary.scalar(name, value, step=iteration)
    metric.reset_states()
    return value.numpy()


