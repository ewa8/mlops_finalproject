"""
Credit to: https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy
"""
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):

    def __init__(self, train_data_dir: str = '', test_data_dir: str = '', batch_size: int = 100):
        super().__init__()
        self.train_data = torch.load(train_data_dir)
        self.test_data = torch.load(test_data_dir)


    def setup(self, stage = None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)



