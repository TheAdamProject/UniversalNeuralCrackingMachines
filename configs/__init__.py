import tensorflow as tf
import os

import UNCM
from utils import *

DS_TRAIN_p, DS_VAL_p = 'train/*.txt', 'clean_val/*.txt'

hparams = {
    
    'filters_dataset' : None,
    'conditional' : False,
    
    'dataset_dir_home' : "# PATH NEEDED HERE",
    'log_dir_home' :  "# PATH NEEDED HERE",
    'keras_model_home' :  './keras_models',

    'max_len_passwd' : 32 + 1,
    'batch_size' : 64,
    'virtual_batch_size_acc' : 1,
    
    'max_number_epochs' : 100,
    'test_num_steps' : 4096 * 8, # -1 for the whole test-set
    'es_patience': 5,
    
    # when to evaluate on validation set
    'evaluation_freq' : 50000,

    'log_freq' : 512,
    'buffer_size' : 10000,
    
    'opt' : lambda: tf.keras.optimizers.Adam(),
    'model_class' : UNCM,
    
    'lr_deacy_factor' : None,
}

# password model arch
hparams['decoder_arch'] = {
    'char_embedding_psswd' : 512,
    'number_of_rnn_layers' : 3,
    'rnn_size' : 1000,
    
    'recurrent_type' : 1,
    'dense_type' : 1,
    'dense_size' : 512,
}


char_id_table, D = make_vocab_passwords()
hparams['passwd_vocab_size'] = len(D)
hparams['char_id_table'] = char_id_table
hparams['D'] = D

hparams['max_len_username'] = hparams['max_len_passwd']

hparams['email_vocab_path'] = ['./configs/vocabs/D0.txt', './configs/vocabs/D1.txt']

hparams['char_username_vocab'] = make_vocab_passwords(is_username=True)[0]
hparams['d0_vocab'], v_d0 = make_vocab(hparams['email_vocab_path'][0])
hparams['d1_vocab'], v_d1 = make_vocab(hparams['email_vocab_path'][1])

hparams['train_ds_dir'] = os.path.join(hparams['dataset_dir_home'], DS_TRAIN_p)
hparams['val_ds_dir'] = os.path.join(hparams['dataset_dir_home'], DS_VAL_p)

# params for testing

thparams = {
    'theta_size' : 100000,
    'decoder_batch_size' : 2048,
}


hparams['testing'] = thparams