from pytorch_lightning import callbacks
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.data.data_module import DataModule
from src.models.model import TumorClassifier

import argparse
import sys
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial):
    
    # Hyperparameters that is optimized
    batch_size = trial.suggest_int('batch_size', 12, 64)
    dropout    = trial.suggest_float('dropout', 0.2, 0.5)
    output_dim = trial.suggest_int('output_dim', 12, 128)
    
    model = TumorClassifier(dropout=dropout, output_dim=output_dim)
    data  = DataModule(data_dir='../../brain_tumor_dataset/processed', batch_size=batch_size)
    
    trainer = pl.Trainer(
        logger=True,
        max_epochs=5,
        # log_every_n_steps=2,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_acc')],
        gpus=1 if torch.cuda.is_available() else 0,
        )
    
    hparams = {  # hyperparameters
        'batch_size': batch_size,
        'dropout':    dropout,
        'output_dim': output_dim,
    }
    
    trainer.logger.log_hyperparams(hparams)
    trainer.fit(model, datamodule=data)
    
    return trainer.callback_metrics['val_acc'].item()

if __name__ == '__main__':
    
    seed = 42
    torch.manual_seed(seed)
    
    print('Finding optimal hyperparameters using Optuna...')
    parser = argparse.ArgumentParser(description='Optuna arguments')
    parser.add_argument(
        '--pruning', 
        default=False
        )
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[1:])
    print(args)
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)    
    
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=50)
    
    print('Best trial:')
    trial = study.best_trial

    print(f'\t Value: {trial.value}')
    print(f'\t Params:')
    for key, value in trial.params.items():
        print(f'\t \t {key}: {value}')
