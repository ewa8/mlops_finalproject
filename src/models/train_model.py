from pytorch_lightning import callbacks
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from azureml.core import Run
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.data_module import DataModule

import argparse
import sys

from src.models.model import TumorClassifier

import logging
import hydra
from omegaconf import OmegaConf
import os

# log = logging.getLogger(__name__)

@hydra.main(config_path='conf', config_name='config')
def train(cfg):
    print(f'Configuration \n {OmegaConf.to_yaml(cfg)}')
    
    print('Training model')
    # parser = argparse.ArgumentParser(description='Training arguments')
    # parser.add_argument('--use_wandb', default=False)
    # # add any additional argument that you want
    # args = parser.parse_args(sys.argv[1:])
    # print(args)
    
    # run =  Run.get_context()
    # mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri()

    # mlf_logger = MLFlowLogger(experiment_name=run.experiment.name, tracking_uri=mlflow_url)
    # mlf_logger._run_id = run.id

    batch_size = cfg.get('batch_size')
    dropout    = cfg.get('dropout')
    output_dim = cfg.get('output_dim')
    seed       = cfg.get('seed')
    
    torch.manual_seed(seed)
    
    # Use wandb logger
    print('Using wandb...')
    kwargs = {'entity': 'sems'}
    chosen_logger = WandbLogger(project='final_project', **kwargs)

    
    model = TumorClassifier(output_dim=output_dim, dropout=dropout)
    data  = DataModule(data_dir='../../../brain_tumor_dataset/processed', batch_size=batch_size)  # ../ three times, as Hydra changes currrent working direc.
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='../../../src/models/checkpoints',
        filename='tumorclassfier_{epoch:02d}_{val_acc:.2f}',
        mode='max',
    )
    
    trainer = pl.Trainer(
        logger=chosen_logger,
        max_epochs=200, 
        log_every_n_steps=10,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback]
        )
    
    trainer.fit(model, datamodule=data)
    print(checkpoint_callback.best_model_path)

if __name__ == '__main__':
    train()

