#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import torch
import torchmetrics
from metrics.utilities import *
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import ipdb
import sys
import copy
import math

class MtlPretrainSarhanFormat(pl.LightningModule):
    def __init__(self, in_dims, num_layers, layer_dims, feature_transformer, threshold_name, clf_input_type, pretraining_epochs=500, results_dir='./', lr=1e-4, clf_adadelta_lr=1.0, tuning_lr=0.001, lambda_filter=1.0, dropout_rate=0.0, in_hidden_dim=15):
        super().__init__()

        dropout_rate = 0.0
        self.save_hyperparameters()
        self.in_dims = in_dims
        self.metrics = dict()

        layer_diff = math.ceil((in_dims - in_hidden_dim) / 6)
        self.s_threshold = None
        self.feature_transformer = feature_transformer
        self.threshold_options = None
        self.threshold_names = None
        self.threshold_name = threshold_name
        self.clf_input_type = clf_input_type
        self.results_dir = results_dir
        self.lambda_filter = lambda_filter
        self.num_pretraining_epochs = pretraining_epochs

        ###
        # Autoencoder
        ###

        layer1_out = in_dims - layer_diff
        layer2_out = layer1_out - layer_diff
        layer3_out = layer2_out - layer_diff
        layer4_out = layer3_out - layer_diff
        layer5_out = layer4_out - layer_diff
        layer6_out = in_hidden_dim

        print(f'Layer Dims:  {in_dims}, {layer1_out}, {layer2_out}, {layer3_out}, {layer4_out}, {layer5_out}, {in_hidden_dim}')

        self.encoder = nn.Sequential(
            nn.Linear(in_dims, layer1_out),
            nn.ReLU(),
            nn.Linear(layer1_out, layer2_out),
            nn.ReLU(),
            nn.Linear(layer2_out, layer3_out),
            nn.ReLU(),
            nn.Linear(layer3_out, layer4_out),
            nn.ReLU(),
            nn.Linear(layer4_out, layer5_out),
            nn.ReLU(),
            nn.Linear(layer5_out, in_hidden_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_hidden_dim, layer5_out),
            nn.ReLU(),
            nn.Linear(layer5_out, layer4_out),
            nn.ReLU(),
            nn.Linear(layer4_out, layer3_out),
            nn.ReLU(),
            nn.Linear(layer3_out, layer2_out),
            nn.ReLU(),
            nn.Linear(layer2_out, layer1_out),
            nn.ReLU(),
            nn.Linear(layer1_out, in_dims),
        )

        ###
        # MLP
        ###

        hidden_layer_size = 10
        if self.clf_input_type.startswith('Xps'):
            mlp_in_dims = in_dims * 3
        elif len(self.clf_input_type) == 1:
            mlp_in_dims = in_dims
        elif len(self.clf_input_type) == 2:
            mlp_in_dims = in_dims * 2
        elif len(self.clf_input_type) == 3:
            mlp_in_dims = in_dims * 3

        self.mlp_layers = nn.Sequential(
            nn.Linear(mlp_in_dims, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1),
            # Sigmoid layer applied in loss function
        )

    def get_callbacks(self, num_epochs):
        early_stopping = EarlyStopping(
            monitor='train/loss',
            min_delta=1e-5,
            patience=25,
            mode='min',
            verbose=True,
            stopping_threshold=1e-9,
            check_on_train_epoch_end=False
        )

        return []

    # Add in relevant parameters needed for RAEDE
    def setup(self, stage=None):
        if not stage == 'test':
            self.trainer.datamodule.ds_train.original_X = copy.deepcopy(self.trainer.datamodule.ds_train.data_df)
            self.early_stopping_error = 1e-9
            self.class_weights = self.trainer.datamodule.ds_train.class_weights

            # Weight our classes
            label = torch.tensor(self.trainer.datamodule.ds_train.data_df.iloc[:,-1].values).float().to(self.device)
            self.num_attacks = label.sum()
            self.num_benign = label.shape[0] - self.num_attacks

    def forward(self, X, L=None, S=None):
        # Perform reconstruction
        if L == None:
            encoded_x = self.encoder(X)
            L = self.decoder(encoded_x)
            S = X - L

        clf_input = X
        dim = len(X.size()) - 1

        clf_input_sets = list()
        for c in self.clf_input_type:
            if c == 'X':
                clf_input_sets.append(X)
            elif c == 'L':
                clf_input_sets.append(L)
            elif c == 'S':
                clf_input_sets.append(S)
            elif c == '0':
                clf_input_sets.append(torch.zeros_like(X))
            elif c == '1':
                clf_input_sets.append(torch.ones_like(X))

        if 'Xps' in self.clf_input_type:
            clf_input = torch.cat((X + S, torch.ones_like(X), torch.ones_like(X)))
        elif len(clf_input_sets) == 1:
            clf_input = clf_input_sets[0]
        elif len(clf_input_sets) == 2:
            clf_input = torch.cat((clf_input_sets[0], clf_input_sets[1]), dim=dim)
        elif len(clf_input_sets) == 3:
            clf_input = torch.cat((clf_input_sets[0], clf_input_sets[1], clf_input_sets[2]), dim=dim)

        ## St
        #St = None
        #if self.s_threshold:
        #    St, _ = self.feature_transformer.torch_transform(None, self, X, None, self.s_threshold)
        #    if self.clf_input_type == 'St':
        #        clf_input = St

        # Perform classification
        P = self.mlp_layers(clf_input)
        predictions = torch.where(P >= 0.0, torch.ones_like(P), torch.zeros_like(P))

        # Return all outputs
        return P, torch.tensor(predictions, dtype=int), L, S, None #St
        
    ###
    # Return an optimizer to use for  pretraining and one for
    # full loop training with reduced learning rate for AE
    ###
    def configure_optimizers(self):
        pretraining_optimizer = torch.optim.Adadelta([
            {'params':self.encoder.parameters()}, 
            {'params':self.decoder.parameters()}
            ], lr=float(self.hparams['lr']))

        full_training_optimizer = torch.optim.Adam(self.mlp_layers.parameters(), lr=float(self.hparams['clf_adadelta_lr']))

        return [pretraining_optimizer, full_training_optimizer]

    ###
    # Skip classifier training during pretraining and skip
    # pretraining optimizer after that point
    ###
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,):

        if self.current_epoch < self.num_pretraining_epochs:
            if optimizer_idx == 0:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()
        else:
            if optimizer_idx == 1:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()

    def loss_function(self, recon_x, x, y_pred, y):
        if self.current_epoch < self.num_pretraining_epochs:
            return self.ae_loss(recon_x, x, y)
        else:
            return self.clf_loss(y_pred, y)

    def clf_loss(self, y_pred, y):
        return F.binary_cross_entropy_with_logits(y_pred, y.unsqueeze(1))

    def ae_loss(self, recon_x, x, y):
        return F.mse_loss(recon_x, x)

    def training_step(self, batch_dc, batch_idx, optimizer_idx):
        x, L_cache, S_cache, label = batch_dc
        L_cache = copy.deepcopy(L_cache)
        S_cache = copy.deepcopy(S_cache)
        label = copy.deepcopy(label)

        # During pretraining, only execute AE and
        # train on only benign samples
        if self.current_epoch < self.num_pretraining_epochs:
            if optimizer_idx == 0:
                # Cut batch down to just benign samples
                benign_x = x[label.flatten() == 0.0]
                label = label[label.flatten() == 0.0]

                # Execute encoder and decoder on batch
                encoded = self.encoder(benign_x)
                L = self.decoder(encoded)
                S = benign_x - L
                loss = self.loss_function(L, benign_x, None, label)
                self.log(f'train/{self.threshold_name}/loss', loss, on_step=False, on_epoch=True)
                self.log(f'train/{self.threshold_name}/ae_loss', loss, on_step=False, on_epoch=True)
                return {'loss':loss, 'y_true':label, 'L':L, 'S':S}
        else:
            if optimizer_idx == 1:
                y_pred, predictions, L, S, St = self.forward(x, L_cache, S_cache)
                loss = self.loss_function(L, x, y_pred, label)

                self.log(f'train/{self.threshold_name}/loss', loss, on_step=False, on_epoch=True)
                self.log(f'train/{self.threshold_name}/clf_loss', loss, on_step=False, on_epoch=True)
                return {'loss':loss, 'y_pred':predictions, 'y_true':label, 'L':L, 'S':S, 'St':St}

    def training_epoch_end(self, training_step_outputs):
        if self.current_epoch == self.num_pretraining_epochs - 1:
            # Cache L and S for use as input during classification
            dl = DataLoader(self.trainer.datamodule.ds_train, batch_size=len(self.trainer.datamodule.ds_train), shuffle=False, num_workers=8)
            for X, _, _, y in dl:
                pass

            # Get our L and S for saving
            encoded = self.encoder(X)
            L_cache = self.decoder(encoded)
            S_cache = X - L_cache

            self.trainer.datamodule.ds_train.L_cache = L_cache.detach()
            self.trainer.datamodule.ds_train.S_cache = S_cache.detach()

        self.log_epoch_end(f'classifier/train_{self.threshold_name}/', training_step_outputs)

    def validation_step(self, batch_dc, batch_idx):
        x, L, S, label = batch_dc
        
        if self.current_epoch < self.num_pretraining_epochs:
            # Cut batch down to just benign samples
            benign_x = x[label.flatten() == 0.0]
            label = label[label.flatten() == 0.0]

            # Execute encoder and decoder on batch
            encoded = self.encoder(benign_x)
            L = self.decoder(encoded)
            S = benign_x - L
            loss = self.loss_function(L, benign_x, None, label)
            self.log(f'val/{self.threshold_name}/loss', loss, on_step=False, on_epoch=True)
            self.log(f'val/{self.threshold_name}/ae_loss', loss, on_step=False, on_epoch=True)
            return {'val_loss':loss, 'y_true':label, 'L':L, 'S':S}

        else:
            y_pred, predictions, L, S, St = self.forward(x, L=None, S=None)
            loss = self.loss_function(L, x, y_pred, label)
            self.log(f'val/{self.threshold_name}/loss', loss, on_step=False, on_epoch=True)
            self.log(f'val/{self.threshold_name}/clf_loss', loss, on_step=False, on_epoch=True)
            return {'val_loss':loss, 'y_pred':predictions, 'y_true':label, 'L':L, 'S':S, 'St':St}

    def validation_epoch_end(self, val_step_outputs):
        # NOTE:  This is still using training data here
        X = torch.tensor(self.trainer.datamodule.ds_train.original_X.iloc[:,:-1].values).float().to(self.device)
        labels = self.trainer.datamodule.ds_train.data_df.iloc[:,-1].values

        ####
        ## Calculate threshold options
        ####

        #benign_x = X[labels == 0]
        #attack_x = X[labels == 1]
        #benign_y_pred, benign_predictions, benign_recon, benign_S, benign_St = self.forward(benign_x)
        #benign_loss = self.ae_loss(benign_recon, benign_x, torch.zeros_like(benign_y_pred))
        #self.log("val/benign_loss", benign_loss, on_step=False, on_epoch=True)

        #if attack_x.size()[0] > 0:
        #    attack_y_pred, attack_predictions, attack_recon, attack_S, attack_St = self.forward(attack_x)
        #    attack_loss = self.ae_loss(attack_recon, attack_x, torch.ones_like(attack_y_pred))
        #    self.log("val/attack_loss", attack_loss, on_step=False, on_epoch=True)

        #    # Calculate automated S thresholds
        #    # Only if we have attacks as otherwise we can't figure it out
        #    threshold_options = self.calculate_threshold_options(benign_loss, attack_loss)

        #    # Now set our threshold since we have values
        #    if not self.threshold_name == 'no_threshold':
        #        threshold_index = self.threshold_names.index(self.threshold_name)
        #        self.s_threshold = self.threshold_options[threshold_index]

        ###
        # Log samples of L, S, and LSO
        ###

        y_pred, predictions, L, S, St = self.forward(X)
        L = L.cpu().detach().numpy()

        # Get our original X
        X = self.trainer.datamodule.ds_train.original_X.iloc[:,:-1].values

        # alternating project, now project to S
        self.S = X - L
        self.LSO = L + self.S

        # Save off information about L, S, and L + S
        self.loggables = {'L':L, 'S':self.S, 'LSO':self.LSO, 'label':labels}

        # Log classification results
        self.log_epoch_end(f'classifier/val_{self.threshold_name}/', val_step_outputs)

    def on_train_end(self):
        # Save off the original data
        original_benign = self.trainer.datamodule.ds_train.original_X.loc[self.trainer.datamodule.ds_train.original_X.label == 0].iloc[:,:-1]
        original_benign = original_benign.iloc[0:100,:]
        original_benign.to_csv(f'{self.hparams["results_dir"]}/train_original_benign.csv')

        # Save off validation set reconstruction data 
        ldf = pd.DataFrame(self.loggables['L'], columns=self.trainer.datamodule.ds_train.data_df.columns[:-1])
        ldf['label'] = self.loggables['label']
        ldf.loc[ldf.label == 0].iloc[0:100,:].to_csv(f'{self.hparams["results_dir"]}/train_l_benign.csv', index=False)

        sdf = pd.DataFrame(self.loggables['S'], columns=self.trainer.datamodule.ds_train.data_df.columns[:-1])
        sdf['label'] = self.loggables['label']
        sdf.loc[sdf.label == 0].iloc[0:100,:].to_csv(f'{self.hparams["results_dir"]}/train_s_benign.csv', index=False)

        lsodf = pd.DataFrame(self.loggables['LSO'], columns=self.trainer.datamodule.ds_train.data_df.columns[:-1])
        lsodf['label'] = self.loggables['label']
        lsodf.loc[lsodf.label == 0].iloc[0:100,:].to_csv(f'{self.hparams["results_dir"]}/train_lso_benign.csv', index=False)

        # Plot LSO Comparison
        vmax = self.trainer.datamodule.ds_train.original_X.iloc[:,:-1].max().max()
        s_vmax = sdf.iloc[0:100,:-1].max().max()

        fig, axarr = plt.subplots(3, 1, sharex=True, sharey=True)
        plt.sca(axarr[0])
        
        plt.imshow(original_benign, vmin=-vmax, vmax=vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        plt.xticks([])
        plt.yticks([])
        ax.xaxis.set_label_position('top')
        plt.ylabel('Original')
        plt.xlabel('Benign Network Flows')
        plt.tight_layout()
        
        plt.sca(axarr[1])
        plt.imshow(ldf.loc[ldf.label == 0].iloc[0:100,:-1], vmin=-vmax, vmax=vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.ylabel('L')
        plt.tight_layout()
        
        plt.sca(axarr[2])
        plt.imshow(sdf.loc[sdf.label == 0].iloc[0:100,:-1], vmin=-s_vmax, vmax=s_vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.ylabel('S')
        plt.tight_layout()
        
        # Save plot and close figure
        plt.savefig(f'{self.hparams["results_dir"]}/train_lso_comparison.png')
        wandb.log({'train/lso_comparison':plt})
        plt.close()

        # Perform some comparisons
        X = self.trainer.datamodule.ds_train.original_X.iloc[:,1:-1].values

        ## Are L or S close to zero?
        #zero_df = np.zeros_like(self.loggables['L'])
        #LZ = np.allclose(self.loggables['L'], zero_df, rtol=1e-3, atol=1e-5)
        #SZ = np.allclose(self.loggables['S'], zero_df, rtol=1e-3, atol=1e-5)
        #XLSO = np.allclose(X, self.loggables['LSO'], rtol=1e-2, atol=1e-2)
        #print(f'LS = {LZ}, SZ = {SZ}, XLSO = {XLSO}')
        #print(f'Threshold Options:  {self.threshold_options}')

    def test_step(self, batch_dc, batch_idx):
        x, L, S, label = batch_dc

        y_pred, predictions, L, S, St = self.forward(x, L=None, S=None)
        loss = self.loss_function(L, x, y_pred, label)
        self.log(f'test/{self.threshold_name}/loss', loss, on_step=False, on_epoch=True)
        self.log(f'test/{self.threshold_name}/clf_loss', loss, on_step=False, on_epoch=True)
        return {'test_loss':loss, 'y_pred':predictions, 'y_true':label, 'L':L, 'S':S, 'St':St}

    def test_epoch_end(self, test_step_outputs):
        self.log_epoch_end(f'classifier/test_{self.threshold_name}/', test_step_outputs)

    def calculate_threshold_options(self, benign_loss, attack_loss):
        high_threshold = np.max([benign_loss, attack_loss])
        low_threshold = np.min([benign_loss, attack_loss])
        fifty_percent_threshold = (high_threshold + low_threshold) / 2.
        seventy_five_percent_threshold = (high_threshold + low_threshold) * 0.75
        twenty_five_percent_threshold = (high_threshold + low_threshold) * 0.25

        # Store options for later
        self.threshold_options = [high_threshold, seventy_five_percent_threshold, fifty_percent_threshold, twenty_five_percent_threshold, low_threshold]
        self.threshold_names = ['high', '75_percent', '50_percent', '25_percent', 'low']

        return self.threshold_options

    def on_save_checkpoint(self, checkpoint):
        checkpoint['threshold_options'] = self.threshold_options
        checkpoint['threshold_names'] = self.threshold_names

    def on_load_checkpoint(self, checkpoint):
        self.threshold_options = checkpoint['threshold_options']
        self.threshold_names = checkpoint['threshold_names']

    def log_epoch_end(self, epoch_type, outputs):
        # Obtain and log metrics
        print(f'{epoch_type}:')

        if self.current_epoch < self.num_pretraining_epochs:
            if isinstance(outputs[0], dict):
                truth = torch.tensor(torch.cat([out['y_true'] for out in outputs]), dtype=int).cpu().detach().numpy()

                # Plot our LSO comparison
                L = torch.cat([out['L'] for out in outputs]).squeeze(1).cpu().detach().numpy()
                S = torch.cat([out['S'] for out in outputs]).squeeze(1).cpu().detach().numpy()
            else:
                truth = torch.tensor(torch.cat([out[0]['y_true'] for out in outputs]), dtype=int).cpu().detach().numpy()

                # Plot our LSO comparison
                L = torch.cat([out[0]['L'] for out in outputs]).squeeze(1).cpu().detach().numpy()
                S = torch.cat([out[0]['S'] for out in outputs]).squeeze(1).cpu().detach().numpy()

            X = L + S
            self.plot_lso_comparison(epoch_type, X, L, S, truth)
        else:
            if isinstance(outputs[0], dict):
                truth = torch.tensor(torch.cat([out['y_true'] for out in outputs]), dtype=int).cpu().detach().numpy()

                # Reconstruct y_pred and y_true
                predictions = torch.cat([out['y_pred'] for out in outputs]).squeeze(1).cpu().detach().numpy()

                # Plot our LSO comparison
                L = torch.cat([out['L'] for out in outputs]).squeeze(1).cpu().detach().numpy()
                S = torch.cat([out['S'] for out in outputs]).squeeze(1).cpu().detach().numpy()
            else:
                truth = torch.tensor(torch.cat([out[0]['y_true'] for out in outputs]), dtype=int).cpu().detach().numpy()

                # Reconstruct y_pred and y_true
                predictions = torch.cat([out[0]['y_pred'] for out in outputs]).squeeze(1).cpu().detach().numpy()

                # Plot our LSO comparison
                L = torch.cat([out[0]['L'] for out in outputs]).squeeze(1).cpu().detach().numpy()
                S = torch.cat([out[0]['S'] for out in outputs]).squeeze(1).cpu().detach().numpy()

            X = L + S
            self.plot_lso_comparison(epoch_type, X, L, S, truth)
            
            self.metrics[f'{epoch_type}predictions'] = predictions
            self.metrics[f'{epoch_type}truth'] = truth

            # Collect metrics
            self.metrics[f'{epoch_type}prec'] = precision(
                predicted_labels=predictions,
                true_labels=truth,
                metrics_dir=self.results_dir,
                is_training=False,
                log_to_disk=False
            )
            
            self.metrics[f'{epoch_type}rec'] = recall(
                predicted_labels=predictions,
                true_labels=truth,
                metrics_dir=self.results_dir,
                is_training=False,
                log_to_disk=False
            )
            
            self.metrics[f'{epoch_type}f1'] = f1_score(
                predicted_labels=predictions,
                true_labels=truth,
                metrics_dir=self.results_dir,
                is_training=False,
                log_to_disk=False
            )

            self.metrics[f'{epoch_type}accuracy'] = accuracy(
                predicted_labels=predictions,
                true_labels=truth,
                metrics_dir=self.results_dir,
                is_training=False,
                log_to_disk=False
            )

            # Save confusion matrix
            cm, cm_norm = plot_confusion_matrix(
                predicted_labels=predictions,
                true_labels=truth,
                metrics_dir=self.results_dir,
                is_training=False,
                prefix=epoch_type.replace('/', '_'),
                title=f'{epoch_type} Confusion Matrix',
                target_names=['benign','attack']
            )

            self.metrics[f'{epoch_type}false_alarm_rate'] = false_alarm_rate(
                predicted_labels=predictions,
                true_labels=truth,
                metrics_dir=self.results_dir,
                is_training=False,
                log_to_disk=False
            )
            
            # Log our metrics
            self.log(f'{epoch_type}{self.feature_transformer.__class__.__name__}_precision', self.metrics[f'{epoch_type}prec'], on_step=False, on_epoch=True)
            self.log(f'{epoch_type}{self.feature_transformer.__class__.__name__}_recall', self.metrics[f'{epoch_type}rec'], on_step=False, on_epoch=True)
            self.log(f'{epoch_type}{self.feature_transformer.__class__.__name__}_f1-score', self.metrics[f'{epoch_type}f1'], on_step=False, on_epoch=True)
            self.log(f'{epoch_type}{self.feature_transformer.__class__.__name__}_accuracy', self.metrics[f'{epoch_type}accuracy'], on_step=False, on_epoch=True)
            self.log(f'{epoch_type}{self.feature_transformer.__class__.__name__}_false_alarm_rate', self.metrics[f'{epoch_type}false_alarm_rate'], on_step=False, on_epoch=True)

    def plot_lso_comparison(self, prefix, X, L, S, labels):
        # If we have no attacks, just skip this epoch
        if np.sum(labels) == 0:
            return

        wandb_prefix = prefix
        prefix = prefix.replace('/', '_')
        L = np.hstack((L, labels))
        l_benign = L[L[:,39] == 0]
        l_attack = L[L[:,39] == 1]

        S = np.hstack((S, labels))
        s_benign = S[S[:,39] == 0]
        s_attack = S[S[:,39] == 1]

        X = np.hstack((X, labels))
        x_benign = X[X[:,39] == 0]
        x_attack = X[X[:,39] == 1]

        # Save the actual data
        x_benign_df = pd.DataFrame(x_benign[0:100,:-1])
        x_benign_df.to_csv(f'{self.results_dir}/{prefix}_x_benign.csv', index=False)
        x_attack_df = pd.DataFrame(x_attack[0:100,:-1])
        x_attack_df.to_csv(f'{self.results_dir}/{prefix}_x_attack.csv', index=False)
        l_benign_df = pd.DataFrame(l_benign[0:100,:-1])
        l_benign_df.to_csv(f'{self.results_dir}/{prefix}_l_benign.csv', index=False)
        l_attack_df = pd.DataFrame(l_attack[0:100,:-1])
        l_attack_df.to_csv(f'{self.results_dir}/{prefix}_l_attack.csv', index=False)
        s_benign_df = pd.DataFrame(s_benign[0:100,:-1])
        s_benign_df.to_csv(f'{self.results_dir}/{prefix}_s_benign.csv', index=False)
        s_attack_df = pd.DataFrame(s_attack[0:100,:-1])
        s_attack_df.to_csv(f'{self.results_dir}/{prefix}_s_attack.csv', index=False)

        # Based on the fact that we normalize our data to be between 0 and 1
        vmax = 2
        s_benign_vmax = s_benign[0:100,:-1].max().max()
        s_attack_vmax = s_attack[0:100,:-1].max().max()
        s_vmax = np.max([s_benign_vmax, s_attack_vmax])

        fig, axarr = plt.subplots(3, 2, sharex=True, sharey=True)
        plt.sca(axarr[0,0])
        
        plt.imshow(x_benign[0:100,:-1], vmin=-vmax, vmax=vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        plt.xticks([])
        plt.yticks([])
        ax.xaxis.set_label_position('top')
        plt.ylabel('X')
        plt.xlabel('Benign Network Flows')
        plt.tight_layout()
        
        plt.sca(axarr[0,1])
        plt.imshow(x_attack[0:100,:-1], vmin=-vmax, vmax=vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.yaxis.set_visible(False)
        plt.colorbar()
        ax.xaxis.set_label_position('top')
        plt.xlabel('Attack Network Flows')
        plt.tight_layout()
        
        plt.sca(axarr[1,0])
        plt.imshow(l_benign[0:100,:-1], vmin=-vmax, vmax=vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.ylabel('L')
        plt.tight_layout()
        
        plt.sca(axarr[1,1])
        plt.imshow(l_attack[0:100,:-1], vmin=-vmax, vmax=vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.colorbar()
        plt.tight_layout()
        
        plt.sca(axarr[2,0])
        plt.imshow(s_benign[0:100,:-1], vmin=-s_vmax, vmax=s_vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.ylabel('S')
        plt.tight_layout()
        
        plt.sca(axarr[2,1])
        plt.imshow(s_attack[0:100,:-1], vmin=-s_vmax, vmax=s_vmax, cmap='BrBG', aspect='auto')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.colorbar()
        plt.tight_layout()
        
        # Save plot and close figure
        plt.savefig(f'{self.results_dir}/{prefix}_lso_comparison.png')
        wandb.log({f'{wandb_prefix}lso_comparison':plt})
        plt.close()
