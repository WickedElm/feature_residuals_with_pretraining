#!/usr/bin/env python

import hydra
from omegaconf import DictConfig, OmegaConf 
import argparse
import importlib
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from metrics.utilities import *
from callbacks.sklearn_trainer_callback import SklearnTrainerCallback
import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
import ipdb
import sys
import dataset.Dataset
import glob
import re
import copy
import math
import pickle
import socket

def list_models():
    # Get all files in models directory
    model_files = glob.glob(f'{hydra.utils.get_original_cwd()}/models/*.py')

    print('')
    print('Models Available: (--model)')
    print('')

    # Get all classes in those files
    for mf in model_files:
        if '__init__' in mf:
            continue
        
        # Get module name from file name
        _, file_name = os.path.split(mf)
        file_name = file_name.replace('.py', '')

        # Get agent classes from file
        with open(mf, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if re.match(r'^class', line):
                model_class_name = (re.findall(r'^class\s+(\w+)\(.*', line))[0]
                print(f'        - {file_name}.{model_class_name}')
    print('')

@hydra.main(config_path='conf', config_name='config')
def main(cfg : DictConfig) -> None:
    # Seed everything for distributed processing
    pl.seed_everything()

    # Set up general arguments
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--project', required=True, default='testing', help='Top level project to hold a set of related results.')
    #parser.add_argument('--results_dir', required=True, default='./output', help='Directory to store any saved results.')
    #parser.add_argument('--experiment', required=True, default='default-experiment', help='Name of the experiment being run.')
    #parser.add_argument('--dataset_path', required=True, default=None, help='Dataset to load.')
    #parser.add_argument('--save_prefix', required=False, default='', help='Prefix to use when saving the trained model.')
    #parser.add_argument('--autoencoder_path', required=False, default=None, help='Path to load autoencoder if needed.')
    #parser.add_argument('--feature_type', required=False, default='original', help='Valid options are original|L|S|original_L|original_S|original_LS|LS')
    #parser.add_argument('--model', required=False, default='autoencoder_linear.AutoencoderLinear', help='Class name for model to be used')
    #parser.add_argument('--clf_model', required=False, default=None, type=str, help='If present, use this classifier.')
    #parser.add_argument('--feature_transformer', required=False, default='original_feature_transformer.OriginalFeatureTransformer', type=str, help='Transformer from original features for classifier.')
    #parser.add_argument('--batch_size', required=False, default=32, type=int, help='Batch size to be used during training.')
    #parser.add_argument('--num_epochs', required=False, default=10000, type=int, help='Number of epochs to perform during training.')
    #parser.add_argument('--lr', required=False, default=1, type=float, help='Learning rate to use  during training.')
    #parser.add_argument('--list', required=False, default=False, dest='list', action='store_true', help='List models and exit')
    #parser.add_argument('--early_stopping_error', required=False, default=1e-7, type=float, help='If the RobustAutoencoder stop criteria reaches this threshold we end training.')
    #parser.add_argument('--n_neighbors', required=False, default=10, type=int, help='For KNN classifier, how many neighbors to use.')
    #parser.add_argument('--lambda_filter', required=False, default=0.01, type=float, help='Lambda filter for Robust Autoencoder.')
    #parser.add_argument('--total_rows_threshold', required=False, default=150000, type=int, help='The total number of rows to use from our data distrubuted among our train/val/test splits.')
    #parser.add_argument('--data_module', required=False, default='data_modules.NetflowDataModule', help='Data module to use for loading data to be used by the model.')
    #parser.add_argument('--sklearn', required=False, default=False, dest='sklearn', action='store_true', help='Indicates if classifier is an sklearn model.')
    #parser.add_argument('--l2', required=False, default=0.4, type=float, help='Level of regularization to apply to optimzier in classifier model.')
    #parser.add_argument('--clf_epochs', required=False, default=500, type=int, help='Number of epochs to perform for training a classifier.')
    #parser.add_argument('--rf_max_features', required=False, default='auto', type=str, help='Maximum number of features for each random forest tree to consider.')
    #parser.add_argument('--load_from_disk', required=False, default=False, dest='load_from_disk', action='store_true', help='If provided, we load a static dataset from disk, else, we reprocess the data.')
    #parser.add_argument('--save_data_prefix', required=False, default='conference', help='Prefix added to any saved data splits to allow for grouping data.')
    #parser.add_argument('--reserve_type', required=False, default='rae_lambda_tuning', help='Indicates which saved data split to load [rae_lambda_tuning|rae_classifier_tuning|rae_full_loop_testing]')
    #parser.add_argument('--hidden_layer_size', required=False, default=15, type=int, help='Size of the hidden layer for AE.')
    #parser.add_argument('--use_all_training_data', required=False, default=False, dest='use_all_training_data', action='store_true', help='Indicates if all_training_data should be used for classifiers.')
    #parser.add_argument('--s_threshold', required=False, default='0.01', type=str, help='Comma separated list of threshold values used to filter down S.')
    #parser.add_argument('--group', required=False, default='lewbug', help='Group name used to tie runs together in the wandb UI.')

    #opt = parser.parse_args()

    #print(opt)

    # Avoid too many open files error
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Restrict CPUs used
    torch.set_num_threads(7)

    num_gpus = 0
    if torch.cuda.is_available() and socket.gethostname() != 'system76-pc':
        num_gpus = 1

    if cfg.general.list:
        list_models()
        sys.exit(0)

    results_dir = os.getcwd()

    #if not os.path.exists(opt.results_dir):
    #    os.makedirs(opt.results_dir, exist_ok=True)

    # Get name of dataset
    # globals call is a bit sketchy but it works...
    ds_name = os.path.split(cfg.general.dataset_path)[-1].replace('.pkl', '')
    dm_modulename, dm_classname = cfg.general.data_module.split('.')
    dm_module = importlib.import_module(f'data_modules.{dm_modulename}')
    dm_class = getattr(dm_module, dm_classname)
    dm = dm_class(
        opt=cfg,
        data_path=cfg.general.dataset_path, 
        batch_size=cfg.general.batch_size,
        total_rows_threshold=cfg.general.total_rows_threshold,
        load_from_disk=cfg.general.load_from_disk,
        load_data_path=f'{hydra.utils.get_original_cwd()}/tmp_ds_data',
        prefix=cfg.general.save_data_prefix,
        reserve_type=cfg.general.reserve_type,
        oversampling_multiplier=cfg.general.oversampling_multiplier,
    )

    # wandb init
    wandb.init(project=cfg.general.project, name=f'{cfg.general.experiment}-{ds_name}', group=cfg.general.group)

    # Loggers
    wandb_logger = WandbLogger(project=cfg.general.project, name=f'{cfg.general.experiment}-{ds_name}')
    csv_logger = CSVLogger(results_dir, name=f'{cfg.general.experiment}-{ds_name}') 

    model_modulename, model_classname = cfg.general.model.split('.')
    model_module = importlib.import_module(f'models.{model_modulename}')
    model_class = getattr(model_module, model_classname)
    if 'MultitaskAeMlp' in model_classname or model_classname.startswith('Mtl'):

        transformer_modulename, transformer_classname = cfg.general.feature_transformer.split('.')
        transformer_module = importlib.import_module(f'feature_transformers.{transformer_modulename}')
        transformer_class = getattr(transformer_module, transformer_classname)
        feature_transformer = transformer_class()

        if 'Weighted' in model_classname:
            ae = model_class(
                cfg.general.ae_model_input_dims,
                3,
                [28, 21, 15],
                feature_transformer,
                cfg.general.threshold_name,
                cfg.general.clf_input_type,
                results_dir,
                cfg.general.lr,
                cfg.general.lambda_filter,
                0.5,
                cfg.general.hidden_layer_size,
                cfg.general.weight_multiplier
            )
        elif 'Pretrain' in model_classname and 'Tuning' in model_classname:
            ae = model_class(
                cfg.general.ae_model_input_dims,
                3,
                [28, 21, 15],
                feature_transformer,
                cfg.general.threshold_name,
                cfg.general.clf_input_type,
                cfg.general.pretraining_epochs,
                results_dir,
                cfg.general.lr,
                cfg.general.clf_adadelta_lr,
                cfg.general.tuning_lr,
                cfg.general.lambda_filter,
                0.5,
                cfg.general.hidden_layer_size,
                cfg.general.tuning_criteria
            )
        elif 'Pretrain' in model_classname:
            ae = model_class(
                cfg.general.ae_model_input_dims,
                3,
                [28, 21, 15],
                feature_transformer,
                cfg.general.threshold_name,
                cfg.general.clf_input_type,
                cfg.general.pretraining_epochs,
                results_dir,
                cfg.general.lr,
                cfg.general.clf_adadelta_lr,
                cfg.general.tuning_lr,
                cfg.general.lambda_filter,
                0.5,
                cfg.general.hidden_layer_size
            )
        else:
            ae = model_class(
                cfg.general.ae_model_input_dims,
                3,
                [28, 21, 15],
                feature_transformer,
                cfg.general.threshold_name,
                cfg.general.clf_input_type,
                results_dir,
                cfg.general.lr,
                cfg.general.lambda_filter,
                0.5,
                cfg.general.hidden_layer_size
            )

    elif 'Sarhan' in model_classname:
        transformer_modulename, transformer_classname = cfg.general.feature_transformer.split('.')
        transformer_module = importlib.import_module(f'feature_transformers.{transformer_modulename}')
        transformer_class = getattr(transformer_module, transformer_classname)
        feature_transformer = transformer_class()

        ae = model_class(
            cfg.general.ae_model_input_dims,
            3,
            [28, 21, 15],
            feature_transformer,
            cfg.general.threshold_name,
            cfg.general.clf_input_type,
            cfg.general.pretraining_epochs,
            results_dir,
            cfg.general.lr,
            cfg.general.clf_adadelta_lr,
            cfg.general.tuning_lr,
            cfg.general.lambda_filter,
            0.5,
            cfg.general.hidden_layer_size
        )

    elif cfg.general.data_module == 'NetflowDataModule':
        ae = model_class(
            29,
            3,
            [25, 20, 15],
            cfg.general.lr,
            cfg.general.lambda_filter
        )
    elif cfg.general.data_module == 'NetflowLogColumnsDataModule':
        ae = model_class(
            32,
            3,
            [25, 20, 15],
            results_dir,
            cfg.general.lr,
            cfg.general.lambda_filter
        )
    elif dm_classname == 'NetflowLogAllColumnsDataModule' or dm_classname == 'NetflowLogAllColumnsWeightedDataModule' or dm_classname == 'NetflowConferenceDataModule' or dm_classname == 'NetflowConferenceNoNormalizeDataModule' or dm_classname == 'NetflowConferenceAttackOnlyTrainingDataModule' or  dm_classname == 'NetflowConferenceBenignOnlyTrainingDataModule' or dm_classname == 'NetflowConferenceCICIDS2017BenignOnlyTrainingDataModule' or dm_classname == 'NetflowStaticTestDataModule' or dm_classname == 'NetflowConferenceStandardizeNormalizationDataModule' or dm_classname == 'NetflowConferenceBenignOnlyTrainingAndValidationDataModule':

        if 'robust' in cfg.general.model:
            ae = model_class(
                35,
                3,
                [28, 21, 15],
                results_dir,
                cfg.general.lr,
                cfg.general.lambda_filter,
                0.5
            )
        else:
            ae = model_class(
                35,
                3,
                [28, 21, 15],
                results_dir,
                cfg.general.lr,
                cfg.general.lambda_filter,
                0.5,
                cfg.general.hidden_layer_size
            )

    elif cfg.general.data_module == 'NetflowDropOHEColumnsDataModule':
        ae = model_class(
            13,
            3,
            [11, 9, 7],
            results_dir,
            cfg.general.lr,
            cfg.general.lambda_filter
        )

    ###
    # callbacks
    ###

    checkpoint_callback = ModelCheckpoint(
        dirpath=results_dir, 
        save_top_k=-1, 
        every_n_epochs=50, 
        save_last=True, 
        filename=f'AE-{cfg.general.project}-{cfg.general.experiment}-{ds_name}' + '-{epoch:06d}'
    )

    model_callbacks = ae.get_callbacks(cfg.general.num_epochs)
    all_callbacks = [checkpoint_callback] + model_callbacks

    trainer = pl.Trainer(
        gpus=num_gpus, 
        logger=[wandb_logger, csv_logger], 
        max_epochs=cfg.general.num_epochs, 
        callbacks=all_callbacks,
        progress_bar_refresh_rate=500,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epoch,
    )

    # Train/validate and test RAE
    trainer.fit(ae, datamodule=dm)
    trainer.test(ae, datamodule=dm)

    # Print out info for child processes (hack)
    print(f'CHECKPOINT_DIR:  {results_dir}')

    # Exit here if we are only training a RAE
    if not cfg.general.clf_model:
        wandb.finish()
        sys.exit(0)

    ###
    # CLASSIFIER
    ###

    ae.eval()

    # Create new NN module
    # - Pass in our trained AE
    # - Fit the NN module and test it 
    clf_models = cfg.general.clf_model.split(',')
    for clf_model_name in clf_models:

        if cfg.general.sklearn:
            clf_model_modulename = 'dummy_model'
            clf_model_classname = 'DummyModel'
            clf_model_module = importlib.import_module(f'models.{clf_model_modulename}')
            clf_model_class = getattr(clf_model_module, clf_model_classname)
        else:
            clf_model_modulename, clf_model_classname = clf_model_name.split('.')
            clf_model_module = importlib.import_module(f'models.{clf_model_modulename}')
            clf_model_class = getattr(clf_model_module, clf_model_classname)

        # Train/validate and test classifier
        classifier_callbacks = []
        classifier_epochs = cfg.general.clf_epochs

        if cfg.general.clf_model and cfg.general.sklearn:
            classifier_epochs = 1
            feature_transformers = cfg.general.feature_transformer.split(',')
            for feature_transformer in feature_transformers:
                transformer_modulename, transformer_classname = feature_transformer.split('.')
                transformer_module = importlib.import_module(f'feature_transformers.{transformer_modulename}')
                transformer_class = getattr(transformer_module, transformer_classname)
                # LEWBUG:  May add this back in later
                ## Have a callback for each threshold we are interested in
                #for threshold in cfg.general.s_threshold.split(','):
                #    sklearn_trainer_callback = SklearnTrainerCallback(cfg, clf_model_name, ae, transformer_class(), s_threshold=float(threshold), threshold_name=str(threshold))
                #    classifier_callbacks.append(sklearn_trainer_callback)

                # For our MSE reconstruction classifier, only use the corresponding feature transformer
                # For others, do not use reconstruction transformer
                if 'reconstruction' in clf_model_name:
                    if not 'reconstruction' in transformer_modulename:
                        continue
                if not 'reconstruction' in clf_model_name:
                    if 'reconstruction' in transformer_modulename:
                        continue

                if 'threshold' in transformer_modulename:
                    # Have a callback for automatically calculated thresholds as well
                    for i, threshold in enumerate(ae.threshold_options):
                        if 'feature_threshold' in transformer_modulename:
                            continue

                        sklearn_trainer_callback = SklearnTrainerCallback(cfg, clf_model_name, ae, transformer_class(), s_threshold=[threshold], threshold_name=ae.threshold_names[i])
                        classifier_callbacks.append(sklearn_trainer_callback)

                    # Enumerate feature threshold options and only use them with appropriate feature_transformers
                    for i, threshold in enumerate(ae.feature_threshold_options):
                        if 'feature_threshold' not in transformer_modulename:
                            continue

                        sklearn_trainer_callback = SklearnTrainerCallback(cfg, clf_model_name, ae, transformer_class(), s_threshold=threshold, threshold_name=ae.feature_threshold_names[i])
                        classifier_callbacks.append(sklearn_trainer_callback)
                else:
                    sklearn_trainer_callback = SklearnTrainerCallback(cfg, clf_model_name, ae, transformer_class(), s_threshold=[999], threshold_name='no_threshold')
                    classifier_callbacks.append(sklearn_trainer_callback)


            clf_model = clf_model_class()
        else:
            transformer_modulename, transformer_classname = cfg.general.feature_transformer.split('.')
            transformer_module = importlib.import_module(f'feature_transformers.{transformer_modulename}')
            transformer_class = getattr(transformer_module, transformer_classname)
            feature_transformer = transformer_class()
            sample_x, sample_y = feature_transformer.transform(None, ae, dm.ds_train.data_df)

            clf_checkpoint_callback = ModelCheckpoint(
                dirpath=results_dir, 
                save_top_k=-1, 
                every_n_epochs=50,
                save_last=True, 
                filename=f'CLF-{cfg.general.project}-{cfg.general.experiment}-{ds_name}' + '-{epoch:06d}'
            )

            classifier_callbacks = classifier_callbacks + [clf_checkpoint_callback]

            clf_model = clf_model_class(
                sample_x.shape[1],
                3,
                [128, 128+64, 128+64+64],
                feature_transformer,
                results_dir,
                cfg.general.experiment,
                0.5,
                0.5,
                cfg.general.l2
                #cfg.general.lr
            )
            clf_model.rae = ae

        clf_trainer = pl.Trainer(
            gpus=num_gpus, 
            logger=[wandb_logger, csv_logger], 
            max_epochs=classifier_epochs, 
            callbacks=classifier_callbacks,
            progress_bar_refresh_rate=500,
            default_root_dir=results_dir,
            check_val_every_n_epoch=cfg.general.check_val_every_n_epoch,
        )

        # Adjust data module to use new data for training/validation
        clf_trainer.fit(clf_model, datamodule=dm)
        clf_trainer.test(clf_model, datamodule=dm)

        ###
        # Free up memory
        # - Currently unclear if this part is working
        ###

        print('LEWBUG:  DELETING OBJECTS')
        del clf_trainer
        for callback in classifier_callbacks:
            del callback

if __name__ == '__main__':
    main()
