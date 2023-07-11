import os
import tensorflow as tf
from functools import partial

from input_pipeline import *
from confenc_architeture import *
from pmodel_architeture import make_password_model
from loss import autoregressive_loss
from utils import mkdir

def export_models(hparams, name_run, models):

    home = hparams['keras_model_home']
    directory = os.path.join(home, name_run)
    mkdir(directory)

    conf_encoder, password_model = models
    dpath = os.path.join(directory, 'password_model.h5')
    password_model.save(dpath)
    
    if not conf_encoder is None:
        epath = os.path.join(directory, 'conf_encoder.h5')
        conf_encoder.save(epath)
    
        print(f"Models saved in: {directory}")
    
def import_models(hparams, name_run):

    home = hparams['keras_model_home']
    directory = os.path.join(home, name_run)

    dpath = os.path.join(directory, 'password_model.h5')
    pmodel = tf.keras.models.load_model(dpath)
    
    epath = os.path.join(directory, 'conf_encoder.h5')
    if os.path.isfile(epath):
        cencoder = tf.keras.models.load_model(epath)
    else:
        cencoder = None

    return cencoder, pmodel


def make_sign(hparams):
    return (
        # x passwd
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        # y passwd
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),

        # usernames
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        # d0
        tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
        # d1
        tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
        # query tokens
        tf.TensorSpec(shape=(None, hparams.get('number_dummy_inputs_special_tokens', 1)), dtype=tf.int64),
        )

def make_encoder(hparams):
    
    ahparams = hparams['encoder_arch']
    
    xin_username = tf.keras.layers.Input((None,), dtype=tf.int64, ragged=False)
    xin_d0 = tf.keras.layers.Input(1, dtype=tf.int64)
    xin_d1 = tf.keras.layers.Input(1, dtype=tf.int64)
    
    number_dummy_inputs_special_tokens = hparams.get('number_dummy_inputs_special_tokens', 1)
    seed_tokens_input = tf.keras.layers.Input(number_dummy_inputs_special_tokens, dtype=tf.int64)

    # encode emails
    emails_encoded, emails_inter_embs = email_encoder(xin_username, xin_d0, xin_d1, hparams)
    emails_encoded = emails_encoded[tf.newaxis,:,:]
    k = emails_encoded.shape[-1]
    pub_encoded = emails_encoded
        
    special_tokens_e = tf.keras.layers.Embedding(number_dummy_inputs_special_tokens, k, name='mixing_encoder_seed_token')

    seed_tokens = special_tokens_e(seed_tokens_input)
        
    seed = make_mixing_encoder(seed_tokens, pub_encoded, k, hparams)
    
    model = tf.keras.Model(
        [
            xin_username,
            xin_d0,
            xin_d1,
            seed_tokens_input
        ], 
        seed
    )

    return model

def make_get_input_tensors(hparams):
    
    char_id_table = hparams['char_id_table']
    char_id_table_userame =  hparams['char_username_vocab']
    
    d0_vocab = hparams['d0_vocab']
    d1_vocab = hparams['d1_vocab']
        
    @tf.function(
        input_signature=[{
            'username' : tf.TensorSpec((None,), dtype=tf.string),
            'd0' : tf.TensorSpec((None,), dtype=tf.string),
            'd1' : tf.TensorSpec((None,), dtype=tf.string),
            'password' : tf.TensorSpec((None,), dtype=tf.string),
        }]
    )
    def _parse_input(leak):
        
        password_x, password_y = parse_password_autoreg(leak['password'], char_id_table)
        
        username = parse_string_non_areg(leak['username'], char_id_table_userame)
        
        d0 = d0_vocab.lookup(leak['d0'])
        d1 = d1_vocab.lookup(leak['d1']) 
    
        # create dummy input for query tokens 
        number_dummy_inputs_special_tokens = hparams.get('number_dummy_inputs_special_tokens', 1)
        seed_token_dummy_input = tf.constant([list(range(number_dummy_inputs_special_tokens))], dtype=tf.int64)
        return password_x.to_tensor(), password_y.to_tensor(), username.to_tensor(), d0[:,tf.newaxis], d1[:,tf.newaxis], seed_token_dummy_input

    return _parse_input

def make_train_step(models, trainable_variables, opt, hparams):
    encoder, decoder = models
    
    @tf.function(input_signature=make_sign(hparams))
    def _train_step(x, y, username, d0, d1, query_tokens):
        with tf.GradientTape() as tape:

            if encoder is None:
                decoder_input = x
            else:
                seed = encoder( (username, d0, d1, query_tokens), training=True)

                sub_sample_size = hparams.get('subsample_size_password_model_train', -1)
                if sub_sample_size != -1:
                    x = x[:sub_sample_size]
                    y = y[:sub_sample_size]
                
                sample_size = tf.shape(x)[0]
                seeds = tf.tile(seed, (sample_size, 1, 1))
                
                decoder_input = [x, seeds]

            logits, p = decoder(decoder_input, training=True)
            loss_i = autoregressive_loss(y, logits)
            
        gradients = tape.gradient(loss_i, trainable_variables)

        return gradients, loss_i
    
    return _train_step


def make_eval_step(models, hparams):
    encoder, decoder = models
    
    @tf.function(input_signature=make_sign(hparams))
    def _eval_step(x, y, username, d0, d1, query_tokens):
        if encoder is None:
            decoder_input = x
        else:
            seed = encoder( (username, d0, d1, query_tokens), training=False)
            
            sample_size = tf.shape(x)[0]
            seeds = tf.tile(seed, (sample_size, 1, 1))
            decoder_input = [x, seeds]

        logits, p = decoder(decoder_input, training=False)
        loss_i = autoregressive_loss(y, logits)

        return loss_i, (logits, p)
    return _eval_step


def make_models(hparams, conditional=True):
    trainable_variables = []
    
    if conditional:
        encoder = make_encoder(hparams)
        trainable_variables += encoder.trainable_variables
        print(f'Number parameters conf. encoder: {encoder.count_params():.0e}')
    else:
        encoder = None
        
    decoder = make_password_model(hparams, encoder)
    trainable_variables += decoder.trainable_variables
    print(f'Number parameters password model: {decoder.count_params():.0e}')
    
    models = encoder, decoder
    
    train_step_fn = make_train_step(models, trainable_variables, hparams['opt'], hparams)
    eval_step_fn = make_eval_step(models, hparams)    
    
    get_input_tensors = make_get_input_tensors(hparams)

    return trainable_variables, models, get_input_tensors, train_step_fn, eval_step_fn

