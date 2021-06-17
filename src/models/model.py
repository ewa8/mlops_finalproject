from torch import nn
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch.nn.functional as F
from torchvision import models 


class TumorClassifier(pl.LightningModule):
    def __init__(self):
        super(TumorClassifier, self).__init__()

        # Pre-trained model ResNet18
        self.model_conv = models.resnet18(pretrained=True)

        # freeze the gradients
        for param in self.model_conv.parameters():
            param.requires_grad = False
        
        # TODO MAKE THE CORRECT INPUT SHAPE! 
        self.fc1 = nn.Linear(320, 50)  # input should be: out_channels times image size times image size 
        self.fc2 = nn.Linear(50, 1)
        
        # Add dropout
        self.dropout = nn.Dropout(0.25)
    
    
    def forward(self, x):
        x = F.relu(self.model_conv(x))

        print(x.shape)

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        return F.sigmoid(x)
    
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        images, labels = batch
        p_labels = self(images)
        loss = F.binary_cross_entropy(p_labels, labels)
        acc = (p_labels.max(1).indices == labels).float().mean()
        #self.log({'train/loss': loss}, on_step=False, on_epoch=True)
        #self.log({'train/acc': acc}, on_step=False, on_epoch=True)
        print(loss)
        return loss
    
    
    def training_epoch_end(self, outputs):
        print(outputs)
    
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        p_labels = self(images)
        loss = F.binary_cross_entropy(p_labels, labels)
        acc = (p_labels.max(1).indices == labels).float().mean()
        #self.log({'val/loss': loss})
        #self.log({'val/acc': acc})
        return loss
    
    def test_step(self):
        
        return None
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
