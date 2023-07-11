import tensorflow as tf

def clip_individual_wvalues(values, l2_norm_clip):
    v_norm = tf.norm(values, ord=2, axis=2, keepdims=True) / l2_norm_clip
    v_norm_to_apply = tf.maximum(1., v_norm)
    v_clipped = values / v_norm_to_apply
    return v_clipped


def scaled_dot_product_attention_sigmoid(fq, fk):
    matmul_qk = tf.matmul(fq, fk, transpose_b=True)
    dk = tf.cast(tf.shape(fk)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)    
    attention_weights = tf.nn.sigmoid(scaled_attention_logits)
    return attention_weights

class DP_Attention(tf.keras.layers.Layer):
    """ 
        Attention with a single query vector with value-level-DP.
    """
    
    def __init__(self, k, l2_norm_clip, noise_mul):
        super(DP_Attention, self).__init__()
        
        self.l2_norm_clip = l2_norm_clip
        self.noise_mul = noise_mul
        
        self.wq = tf.keras.layers.Dense(k, name="dense_q")
        self.wk = tf.keras.layers.Dense(k, name="dense_k")        
        self.wv = tf.keras.layers.Dense(k, name="dense_v")
        
    def __call__(self, q, k, v):
        
        """ Code for single query vector """
        assert q.shape[1] == 1
        
        # apply projections
        fq = self.wq(q)
        fk = self.wk(k)
        fv = self.wv(v)
        
        # get weights (sigmoid output)
        attention_weights = scaled_dot_product_attention_sigmoid(fq, fk)
        
        ## DP part0: Clip individual weighted values
        fv = clip_individual_wvalues(fv, self.l2_norm_clip)
        
        # scaling values according attention_weights
        ## Note: as the attention_weights are in [0, 1], applying them after the clipping does not 
        ## increase the norm of the value vectors
        fv_w = fv * tf.transpose(attention_weights, (0, 2, 1))
        
        # sum clipped values (i.e., "weighted" mean)
        seed = tf.reduce_sum(fv_w, 1, keepdims=True)
        
        ## Apply noise proportional to sensitivity
        seed = seed + tf.random.normal(tf.shape(seed), stddev=self.noise_mul*self.l2_norm_clip)
            
        return seed, attention_weights
    




    
