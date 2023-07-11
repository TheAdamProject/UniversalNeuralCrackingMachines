import tensorflow as tf

def autoregressive_loss(y, logits):
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y, logits)

    mask = (y != 0)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ =  loss_ * mask
    # compute seq length for avg
    n = tf.reduce_sum(mask)
    # compute avg
    loss = tf.reduce_sum(loss_) / n
    
    return loss


def get_probability(y, logits, log_probability):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y, logits)
    
    loss = tf.cast(loss, tf.float64)

    mask = (y != 0)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss =  loss * mask
    loss = -tf.reduce_sum(loss, 1) 
    
    if not log_probability:
        loss = tf.exp(loss)
    
    return loss