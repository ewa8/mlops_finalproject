import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = '', batch_size: int = 100):
        super().__init__()
        self.train_data = torch.load(f'{data_dir}/train.pt')
        self.test_data = torch.load(f'{data_dir}/test.pt')
        
        train_dataset = TensorDataset(self.train_data['data'], self.train_data['target'])
        self.test_dataset = TensorDataset(self.test_data['data'], self.test_data['target'])
        
        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [len(train_dataset)-int(len(train_dataset)*0.3), int(len(train_dataset)*0.3)]
            )
        
        self.batch_size = batch_size

    def setup(self, stage = None):
        pass
    
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)


    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)

