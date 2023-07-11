import tensorflow as tf
from tensorflow import keras as k
import numpy as np
import tqdm
import os

import utils
from grad_accum import gradient_scaling, gradient_scaled_sum

class Trainer:
    
    def setup_log_files(self, name_run):
        log_dict = os.path.join(self.hparams['log_dir_home'], name_run)
        print(f"Logs in {log_dict}")
        log_eval_dict = os.path.join(log_dict, 'eval')
        
        if not os.path.isdir(log_dict):
            os.mkdir(log_dict)
            os.mkdir(log_eval_dict)
            
        return log_dict, log_eval_dict
        
        
    def __init__(self,
                 name_run,
                 make_models,
                 ds_train,
                 ds_val,
                 hparams
                ):
        
        self.conditional = hparams['conditional']
        self.trainable_variables, self.models, self.get_input_tensors, self.train_step, self.eval_step = make_models(hparams, conditional=self.conditional)
                
        if not ds_train is None:
            self.ds_train = ds_train.repeat(-1)
        self.ds_val = ds_val
        
        self.opt = hparams['opt']()
        self.hparams = hparams
        
        self._metric_loss_train = tf.keras.metrics.Mean(name='loss')
        self._metric_loss_validation = tf.keras.metrics.Mean(name='loss')
        
        self.best_score_es = None 
        
        self.log_dict_train, self.log_dict_eval = self.setup_log_files(name_run)
        self.train_summary_writer = train_summary_writer = tf.summary.create_file_writer(self.log_dict_train)
        self.eval_summary_writer = test_summary_writer = tf.summary.create_file_writer(self.log_dict_eval)
        
        #check-points
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.opt,
            model=[model for model in self.models if not model is None],
        )

        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, self.log_dict_train, max_to_keep=1)

        last_checkpoint = self.ckpt_manager.latest_checkpoint
        if last_checkpoint:
            self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            print(f'Loading checkpoint: {last_checkpoint}')
                
                
    def export_best_model(self):
        
        last_checkpoint = self.ckpt_manager.latest_checkpoint
        if last_checkpoint:
            self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            print(f'Loading checkpoint: {last_checkpoint}')
        else:
            raise Exception("No saved checkpoint")
        
        for i, model in enumerate(self.models):
            
            if model is None:
                continue
        
            name_model = f'model_{i}.h5'
            path = os.path.join(self.log_dict_train, name_model) 
            print(f"Saving {name_model} in {path} ...")
            model.save(path)

    def train_step_gradient_accumulation(self, ds_train_iter):
        
        v_batch_size = self.hparams['virtual_batch_size_acc']
        avg_gradient = None
        outofrange = False

        for i in range(v_batch_size):
            leak_sample = next(ds_train_iter)

            gradient_i, loss_i = self.train_step(*self.get_input_tensors(leak_sample))
            self._metric_loss_train.update_state(loss_i)

            #accum. gradient
            if i == 0:
                avg_gradient = gradient_scaling(gradient_i, v_batch_size)
            else:
                avg_gradient = gradient_scaled_sum(avg_gradient, gradient_i, v_batch_size)

        self.opt.apply_gradients(zip(avg_gradient, self.trainable_variables))
    
    
    def evaluate(self, iteration_number):
        val = self.ds_val.take(self.hparams['test_num_steps'])
        for leak_sample in val:
            loss_i, model_outputs = self.eval_step(*self.get_input_tensors(leak_sample))
            self._metric_loss_validation.update_state(loss_i)
        score = utils.flush_metric(iteration_number, self._metric_loss_validation, tfb_log=True)
        return score
    
    def early_stopping(self, new_score):

        if self.best_score_es is None or new_score < self.best_score_es:
            print("New Best", new_score)
            self.best_score_es = new_score
            self.countdown = self.hparams['es_patience']
            
            # Save checkpoint
            ckpt_save_path = self.ckpt_manager.save()
            print ('Saving checkpoint at {}'.format(ckpt_save_path))
            
            return False
        
        print("Perfomance decreased:", new_score, self.best_score_es)
        
        # drop learning rate
        self.learningrate_decay()
        
        self.countdown -= 1
        if self.countdown == 0:
            print("Early stop:", new_score, self.best_score_es)
            return True

    
    def is_evalution_time(self, is_new_epoch, iteration_number):        
        if self.hparams['evaluation_freq'] is None:
            return is_new_epoch
        else:
            return iteration_number % self.hparams['evaluation_freq'] == 0
            
    
    def __call__(self):

        ds_train_iter = iter(self.ds_train)
        epochs = 0
        iteration_number = 0
        pbar = tqdm.tqdm()

        while True:
            self.train_step_gradient_accumulation(ds_train_iter)
            is_new_epoch = False

            if iteration_number % self.hparams['log_freq'] == 0:

                with self.train_summary_writer.as_default():
                    loss = utils.flush_metric(iteration_number, self._metric_loss_train, tfb_log=True)
                    print(f"[{iteration_number:07d} (epoch: {epochs})] loss_train: {loss:0.4f}")

            if self.is_evalution_time(is_new_epoch, iteration_number):
                print("Starting evaluation ...")
                with self.eval_summary_writer.as_default():
                    score = self.evaluate(iteration_number)
                is_terminated = self.early_stopping(score)
                print(f"\t[{iteration_number:07d} (epoch: {epochs})] loss_val: {score:0.4f}")

                if is_terminated:
                    break

            if epochs >= self.hparams['max_number_epochs']:
                print("Maximum number of epochs reached!")
                break

            iteration_number += 1
            pbar.update(1)

        pbar.close()
        self.export_best_model()
        
        
    def learningrate_decay(self):
        lr_deacy_factor = self.hparams['lr_deacy_factor']
        if lr_deacy_factor:
            self.opt.learning_rate = self.opt.learning_rate * lr_deacy_factor
            print(f"New learning rate-> {self.opt.learning_rate}")