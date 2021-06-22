import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):

    def __init__(self, train_data_dir: str = '', test_data_dir: str = '', batch_size: int = 100):
        super().__init__()
        self.train_data = torch.load(f'{train_data_dir}/train.pt')
        self.test_data = torch.load(f'{train_data_dir}/test.pt')
        
        self.train_dataset = TensorDataset(self.train_data['data'], self.train_data['target'])
        self.test_dataset = TensorDataset(self.train_data['data'], self.train_data['target'])
        
        self.batch_size = batch_size

    def setup(self, stage = None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)



