from . import *
import UNCM

hparams['model_class'] = UNCM
hparams['conditional'] = True

# training process 
hparams['sample_size'] = 8192 // 4

hparams['virtual_batch_size_acc'] = 16
hparams['batch_size'] = 0

hparams['test_num_steps'] = -1
hparams['log_freq'] = 32

hparams['evaluation_freq'] = 900
hparams['v_size'] = 1024


# configuration encoder arch
hparams['encoder_arch'] = {
    'projecter_type' : 0,
    'mixarch_type' : 1,
    
    # email encoder
    'email_encoder_arch': {
        #'type'
        'char_username_embedding_size': 16,
        'email_rnn_size' : 256,

        'd0_embedding_size' : 256,
        'd1_embedding_size' : 256,        
    },
    
    'psswd_encoder_arch' : {
        'embedding_size' : 512,
        'rnn_size' : 256,
    }
}


hparams['number_dummy_inputs_special_tokens'] = hparams['decoder_arch']['number_of_rnn_layers'] * 2
