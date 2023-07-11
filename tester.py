import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from input_pipeline import read_and_parse_single_leak, make_dataset
from loss import get_probability 
from inference import ancestral_sampling

def scoring(G, n=10):
    Gs = np.sort(G)
    gy = np.cumsum(np.ones(len(Gs))) / len(Gs)
    
    scores = np.zeros(n)
    for i, guess_limit in enumerate(np.linspace(0.1, 1, n)):
        q = (gy<guess_limit).sum()
        num_guesses = Gs[q]
        scores[i] = num_guesses
    
    return scores


class Tester:
    """ Assign guess numbers and/or probabilities to passwords using Monte Carlo est. """
    
    def __init__(self, encoder, decoder, input_fn, hparams, log_probability=False):
        
        self.encoder, self.decoder = encoder, decoder
        self.conditional = not self.encoder is None
        
        self.hparams = hparams.copy()
        self.thparams = hparams['testing']
        self.input_fn = input_fn
        
        if "sample_size" in self.thparams:
            sample_size = self.thparams["sample_size"]
            self.hparams["sample_size"] = sample_size

        self.log_probability = log_probability
        
        #self.setup_theta()
        self.u_p_theta = None
    
    @staticmethod
    def _guess_number(P, theta_P, epsilon=0):
        """ REMEMBER: move to -log p"""

        def gnmc(pt, theta_P):
            i = np.searchsorted(theta_P, pt)
            gn = ( 1 / (theta_P[i:] * len(theta_P) + epsilon) ).sum()
            return gn

        n = len(P)
        G = np.zeros(n)
        for i in range(n):
            G[i] = gnmc(P[i], theta_P)

        return G
    
    @staticmethod
    def plot_guess_number(
        G,
        guessed_limit=1.,
        guesses_limit=(10**1, 10**18),
        ax=None,
        x_lim=None,
        **plot_kargs
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        Gs = np.sort(G)

        gss_limit = (Gs<guesses_limit[1]).sum()

        gy = np.cumsum(np.ones(len(Gs))) / len(Gs)
        ged_limit = (gy<guessed_limit).sum()

        n = min([gss_limit, ged_limit])

        x = Gs[:n]
        y = np.cumsum(np.ones(n) / len(Gs))

        ax.plot(x, y, **plot_kargs)

        ax.set_xscale('log')

        if x_lim is None:
            x_lim = x[-1]
        else:
            x_lim = max((x[-1], x_lim))

        margin = .04
        yticks = np.linspace(0, 1, 11)
        yticklabels = [f'{int(i*100)}%' for i in yticks]
        ax.set(
            xlabel="Number of Guesses (log)",
            ylabel="Guessed Passwords",
            ylim=(0, 1 + margin),
            xlim=(guesses_limit[0], x_lim),
            yticks=yticks,
            yticklabels=yticklabels,
        );

        ax.legend()
        ax.grid()
        return ax, x_lim
    
    
    def setup_theta(self):
        # get theta
        self.hparams['batch_size'] = self.thparams['theta_size']
        ds_for_theta, N = make_dataset(self.hparams['val_ds_dir'], self.hparams, conditional=False)
        self.theta = ds_for_theta.take(1).get_single_element()
        print("Size theta ---> ", len(self.theta['password']))

        self.theta_ds = tf.data.Dataset.from_tensor_slices(self.theta) 
        self.theta_ds = self.theta_ds.apply(
            tf.data.experimental.dense_to_ragged_batch(self.thparams['decoder_batch_size'])
        )


    
    def compute_seed(self, path):
        # get dataset for configuration seed
        ds_sample_for_seed = read_and_parse_single_leak(path, self.hparams, True, shuffle=True)
        
         # compute seed from partial sample
        _data_for_seed = ds_sample_for_seed.get_single_element()  
        data_for_seed = self.input_fn(_data_for_seed)
                
        print("Actual number of users sampled for SEED computation: ", data_for_seed[0].shape[0], "\n")
                
        inputs = list(data_for_seed[2:])
        
        seed = self.encoder(inputs, training=False)
    
        return seed
        
        
    
    def compute_probability_from_file(self, path, return_X=False):
        
        # get dataset to test
        ds_full = read_and_parse_single_leak(path, self.hparams, False, shuffle=True)
        ds_full = ds_full.apply(
            tf.data.experimental.dense_to_ragged_batch(self.thparams['decoder_batch_size'])
        )
        
        if self.conditional:
            # compute seed 
            seed, *pub_encoded = self.compute_seed(path)
        else:
            seed = None
            pub_encoded = None

        return self.compute_probability(
            seed,
            ds_full,
            return_X=return_X
        ), seed, pub_encoded
        
    
    def compute_probability(self, seed, ds_full, return_X=False):
    
        P = []
        
        if return_X:
            X = []
            
        for _batch in ds_full:
            
            batch = self.input_fn(_batch)
            x, y, *_ = batch
            
            if self.conditional:
                seeds = tf.tile(seed, (x.shape[0], 1, 1))
                logits, *_ = self.decoder((x, seeds), training=False)
            else:                
                logits, _ = self.decoder(x, training=False)

            _P = get_probability(y, logits, log_probability=self.log_probability)
            P.append(_P.numpy())

            if return_X:
                X += [s.decode() for s in _batch['password'].numpy()]
                
        P = np.concatenate(P)
        
        if return_X:
            return X, P
        else:
            return P
    
    
    def assing_probability_theta(self, seed=None):
        
        return self.compute_probability(
            seed,
            self.theta_ds,
        )
    
    def compute_guess_numbers_from_file(self, path):

        (X, P), seed, pub_encoded = self.compute_probability_from_file(path, return_X=True)

        if not self.conditional and self.u_p_theta is None:
            # sample only the first time
            _, self.u_p_theta = ancestral_sampling(
                self.thparams['theta_size'],
                self.decoder,
                self.thparams['decoder_batch_size'],
                self.hparams,
                seed=seed,
                with_string=False
            )
            self.u_p_theta.sort()  
        p_theta = self.u_p_theta
        
        if self.conditional:
            # sample only the first time
            _, p_theta = ancestral_sampling(
                self.thparams['theta_size'],
                self.decoder,
                self.thparams['decoder_batch_size'],
                self.hparams,
                seed=seed,
                with_string=False
            )
            p_theta.sort()

        G = self._guess_number(P, p_theta)

        return X, G, P, seed, pub_encoded

    