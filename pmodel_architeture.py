import tensorflow as tf

def dedicated_seed_dense_lstm(emb, seed, hparams, conditional):
    ahparams = hparams['decoder_arch']
    num_layers = ahparams['number_of_rnn_layers']
    lstm_states = []
    
    if conditional:
        num_queries = seed.shape[1]
        assert num_queries % 2 == 0  
        assert (num_queries / 2) == num_layers    
    
        for i in range(0, num_queries, 2):
            seed_i0 = tf.keras.layers.Dense(ahparams['rnn_size'], use_bias=False, name=f'decoder_seed{i}')(seed[:, i, :])
            seed_i1 = tf.keras.layers.Dense(ahparams['rnn_size'], use_bias=False, name=f'decoder_seed{i+1}')(seed[:, i+1, :])
            lstm_states.append( (seed_i0, seed_i1) )
    else:
        for i in range(num_layers):
            lstm_states.append( None )
    
    x = emb
    for i in range(ahparams['number_of_rnn_layers']):
        x = tf.keras.layers.LSTM(ahparams['rnn_size'], return_sequences=True, name=f'decoder_lstm{i}')(x, initial_state=lstm_states[i])
        
    return x


def single_seed_dense_lstm(emb, seed, hparams, conditional):
    ahparams = hparams['decoder_arch']
    num_layers = ahparams['number_of_rnn_layers']
    lstm_states = []
    
    if conditional:
        assert seed.shape[1] == 1
    
        for i in range(0, num_layers):
            seed_i0 = tf.keras.layers.Dense(ahparams['rnn_size'], use_bias=False, name=f'decoder_seeda{i}')(seed[:, 0, :])
            seed_i1 = tf.keras.layers.Dense(ahparams['rnn_size'], use_bias=False, name=f'decoder_seedb{i}')(seed[:, 0, :])
            lstm_states.append( (seed_i0, seed_i1) )
    else:
        for i in range(num_layers):
            lstm_states.append( None )
    
    x = emb
    for i in range(ahparams['number_of_rnn_layers']):
        x = tf.keras.layers.LSTM(ahparams['rnn_size'], return_sequences=True, name=f'decoder_lstm{i}')(x, initial_state=lstm_states[i])
        
    return x

def make_password_model(hparams, encoder):
    
    ahparams = hparams['decoder_arch']
    vocab_size = hparams['passwd_vocab_size']
    
    conditional = not encoder is None
    
    xin = tf.keras.layers.Input((None,), dtype=tf.int64, ragged=False)
    inputs = [xin]
    
    if conditional:
        seed = tf.keras.layers.Input(encoder.outputs[0].shape[-2:], dtype=tf.float32)
        inputs.append(seed)
    else:
        seed = None
    
    # input embedding
    emb = tf.keras.layers.Embedding(vocab_size, ahparams['char_embedding_psswd'], mask_zero=False, name='decoder_emb')(xin)   

    # recurrent part
    print(f"Password model recurrent type {ahparams['recurrent_type']}")
    
    recurrent_type = ahparams['recurrent_type']
    if recurrent_type == 1:
        x = dedicated_seed_dense_lstm(emb, seed, hparams, conditional)
    elif recurrent_type == 2:
        x = single_seed_dense_lstm(emb, seed, hparams, conditional)
    else:
        raise Exception()
        
    
    # final dense part
    if ahparams['dense_type'] == 1:
        print(f"Password model dense type {ahparams['dense_type']} - {ahparams['dense_size']}")
        x = tf.keras.layers.Dense(ahparams['dense_size'], name='decoder_dense')(x)
    
    logits = tf.keras.layers.Dense(vocab_size, name='decoder_dense_final')(x)
    p = tf.nn.softmax(logits)
    
    return tf.keras.Model(inputs, [logits, p])
