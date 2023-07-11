import tensorflow as tf

from DP_attention import DP_Attention

def make_mixing_encoder(query_tokens, pub_encoded, k, hparams):
    arch_type = hparams['encoder_arch']['mixarch_type']
    
    if arch_type == 1:
        print(f"Non private")
        # MHA with single head
        x = tf.keras.layers.MultiHeadAttention(1, k, name=f'mixing_encoder')(query_tokens, pub_encoded)
    elif arch_type == 10:
        print(f"With DP {hparams['DP_params']}")
        l2_norm_clip, noise_multiplier = hparams['DP_params']
        att = DP_Attention(k, l2_norm_clip, noise_multiplier)
        x, att_weights = att(query_tokens, pub_encoded, pub_encoded)
    return x


def email_encoder(xin_username, xin_d0, xin_d1, hparams):
    ahparams = hparams['encoder_arch']['email_encoder_arch']
    
    e_username = tf.keras.layers.Embedding(
        hparams['char_username_vocab'].size().numpy(),
        ahparams['char_username_embedding_size'],
        mask_zero=True,
    )(xin_username)
    
    e_d0 = tf.keras.layers.Embedding(
        hparams['d0_vocab'].size().numpy(),
        ahparams['d0_embedding_size'],
        mask_zero=False,
    )(xin_d0)
    assert e_d0.shape[1] == 1
    e_d0 = tf.reshape(e_d0, (-1, e_d0.shape[-1]))
    
    e_d1 = tf.keras.layers.Embedding(
        hparams['d1_vocab'].size().numpy(),
        ahparams['d1_embedding_size'],
        mask_zero=False,
    )(xin_d1)
    assert e_d1.shape[1] == 1
    e_d1 = tf.reshape(e_d1, (-1, e_d1.shape[-1]))
    
    enc_username = tf.keras.layers.GRU(
        ahparams['email_rnn_size'],
        return_sequences=False,
    )(e_username)
    
    enc_email = tf.keras.layers.Concatenate(axis=1)([enc_username, e_d0, e_d1])
    
    if hparams['encoder_arch']['projecter_type'] == 1:
        print("with projected enc_email")
        v_size = hparams['v_size']
        enc_email = tf.keras.layers.Activation("tanh")(enc_email)       
        enc_email = tf.keras.layers.Dense(v_size, activation="tanh", use_bias=False, name='email_encoder_final_projection')(enc_email)
    elif hparams['encoder_arch']['projecter_type'] == 2:
        print("with projected enc_email relu")
        v_size = hparams['v_size']
        enc_email = tf.keras.layers.Activation("relu")(enc_email)       
        enc_email = tf.keras.layers.Dense(v_size, activation="relu", use_bias=False, name='email_encoder_final_projection')(enc_email)
    
    return enc_email, (e_username, enc_username, e_d0, e_d1)