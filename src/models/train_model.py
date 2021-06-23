import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from azureml.core import Run
from pytorch_lightning.loggers import MLFlowLogger

from src.data.data_module import DataModule

import argparse
import sys

from src.models.model import TumorClassifier

def train():
    print('Training model')
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--use_wandb', default=False)
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[1:])
    print(args)

    run =  Run.get_context()
    mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri()

    mlf_logger = MLFlowLogger(experiment_name=run.experiment.name, tracking_uri=mlflow_url)
    mlf_logger._run_id = run.id

    if args.use_wandb:  # Use wandb logger
        print('Using wandb...')
        kwargs = {'entity': 'sems'}
        chosen_logger = WandbLogger(project='final_project', **kwargs)
    else:  # Use default logger
        chosen_logger = True
    
    model = TumorClassifier()
    data = DataModule(data_dir='brain_tumor_dataset/processed', batch_size=24)
    
    trainer = pl.Trainer(
        logger=chosen_logger, 
        max_epochs=10, 
        log_every_n_steps=2,
        gpus=1 if torch.cuda.is_available() else 0,
        )
    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    train()

