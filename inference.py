import numpy as np
import tensorflow as tf
import tqdm
import math

def to_string(ids, D):
    s = []
    for i in ids:
        if i == 2:
            break
        s.append(D[i])
    try:
        return ''.join(s)
    except:
        return ''
    
    
def tot_probability(ids, PP):
    assert len(PP) == len(ids)
    P = 1.
    for i in range(len(PP)):
        P *= PP[i]
        if ids[i] == 2:
            break
    return P

def ancestral_sampling_batch(
    model,
    batch_size,
    hparams,
    with_string=True,
    seed=None,
):
    # Extremely inefficient code follows:
    
    max_len = hparams['max_len_passwd']
    
    S = []
    PP = []
    
    init_state = np.zeros((batch_size, max_len), np.int64)
    init_state[:, 0] += 1
    x = tf.Variable(init_state)
    
    if len(model.inputs) == 2 and seed is None:
        raise Exception("Conditional password model cannot have input seed 'None'")
    
    if not seed is None:
        seeds = tf.tile(seed, [batch_size, 1, 1])

    for t in range(max_len-1):
        inputs = [x]
        if not seed is None:
            inputs += [seeds]
            
        logits, p = model(inputs)
        
        predicted_id = tf.random.categorical(
            logits[:,t], 1, dtype=None, seed=None, name=None
        )[:,0]
        
        # update input
        x[:,t+1].assign(predicted_id)    
        
        # get probabilities selected tokens
        pp = tf.gather(p[:, t, :], predicted_id, batch_dims=1, axis=1)
        
        PP.append(pp.numpy()[:, np.newaxis])
        S.append(predicted_id.numpy()[:, np.newaxis])
          

    S = np.concatenate(S, 1)
    if with_string:
        final_S = [to_string(s, hparams['D']) for s in S]
    else:
        final_S = None
        
    PP = np.concatenate(PP, 1)
    final_P = [tot_probability(s, p) for p, s in zip(PP, S)]

    return final_S, final_P


def ancestral_sampling(
    N,
    model,
    batch_size,
    hparams,
    with_string=True,
    seed=None,
):
    num_batches = math.ceil(N / batch_size) 
    
    P, S = [], []
    for i in tqdm.trange(num_batches):
        s, p = ancestral_sampling_batch(
            model,
            batch_size,
            hparams,
            with_string=with_string,
            seed=seed,
        )
        P.append(p)
        if with_string:
            S += s
    P = np.concatenate(P)
    return S, P
        