import tensorflow as tf
from functools import partial
from glob import glob
import tqdm
import numpy as np


def _f_filter(path, TD1, TT):
    p = path.split('/')[-1]
    try:
        name, _, t = p.split('__')

        d1 = name.split('.')[-1]
        t = t.split('.')[0]
    except:
        return False

    return (not TD1 or d1 == TD1) and (not TT or t == TT)

def parse_password_autoreg(x, char_id_table):
    n = tf.shape(x)[0]
    x = tf.strings.bytes_split(x) 
    x = char_id_table.lookup(x)
    
    special_token_x = tf.cast(tf.tile(tf.fill((1,1), 1), (n, 1)), x.dtype)
    special_token_y = tf.cast(tf.tile(tf.fill((1,1), 2), (n, 1)), x.dtype)    

    x_idx = tf.concat((special_token_x, x), 1)
    y_idx = tf.concat((x, special_token_y), 1)

    return x_idx, y_idx


def parse_string_non_areg(x, char_id_table):
    n = tf.shape(x)[0]
    x = tf.strings.bytes_split(x)    
    
    x = char_id_table.lookup(x)
    special_token_y = tf.cast(tf.tile(tf.fill((1,1), 2), (n, 1)), x.dtype)
    x = tf.concat((x, special_token_y), 1)
    
    return x




def read_and_parse_single_leak(path, hparams, conditional, shuffle=True, with_passwords=True):
   
    char_id_table = hparams['char_id_table']

    @tf.function
    def parse_string(s):
        ss = tf.strings.split(s, '\t')
        
        username, d0, d1 = ss[0], ss[1], ss[2]
        
        if with_passwords:
            password =  ss[3]
        else:
            password = ''
        
        username = tf.strings.lower(username)
        d0 = tf.strings.lower(d0)
        d1 = tf.strings.lower(d1)

        return {
            'username' : username,
            'd0' : d0,
            'd1' : d1,
            'password' : password,
        }
        
    # read from disk
    ds = tf.data.TextLineDataset(path)
    
    if shuffle:
        # shuffle
        ds = ds.shuffle(hparams['buffer_size'])   
    
    if conditional:
        # take
        ds = ds.take(hparams['sample_size'])
        
    # split and parse
    ds = ds.map(parse_string)
    
    if conditional:
        ds = ds.apply(
            tf.data.experimental.dense_to_ragged_batch(hparams['sample_size'])
        )
 
    return ds

def make_dataset(home, hparams, conditional=True, filters=None):
    
    if filters:
        print(f"With filters: {filters}")
        f_filter = partial(_f_filter, TD1=filters[0], TT=filters[1])
        home = glob(home)
        home = list(filter(f_filter, home))
    
    files = tf.data.Dataset.list_files(home)
    
    N = files.cardinality()
    
    files = files.shuffle(hparams['buffer_size'])

    _read_and_parse_single_leak = partial(read_and_parse_single_leak, hparams=hparams, conditional=conditional)
    
    if conditional:
        ds = files.flat_map(_read_and_parse_single_leak)
        if hparams['batch_size']:
            ds = ds.batch(hparams['batch_size'])
    else:
        # mixing different leakes (round-robin dispatcher)
        ds = files.interleave(
            _read_and_parse_single_leak,
            block_length=1,
        )
        
        ds = ds.apply(
            tf.data.experimental.dense_to_ragged_batch(hparams['batch_size'])
        )

    return ds, N

