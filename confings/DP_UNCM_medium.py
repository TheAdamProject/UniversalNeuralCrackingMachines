from .UNCM_medium import *

# clipping sensitivity & noise mul 
hparams['DP_params'] = (1., 3.)

# Use Attention DP
hparams['encoder_arch']['mixarch_type'] = 10

# with a single query vector
hparams['decoder_arch']['recurrent_type'] = 2
hparams['number_dummy_inputs_special_tokens'] = 1

