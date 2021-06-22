import torch
from torch import nn
from sklearn.metrics import accuracy_score
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch.nn.functional as F
from torchvision import models 


class TumorClassifier(pl.LightningModule):
    def __init__(self, dropout: float=0.25, output_dim: int=64):
        super(TumorClassifier, self).__init__()

        # Pre-trained model ResNet18
        self.model_conv = models.resnet18(pretrained=True)
        self.model_conv.fc = nn.Identity()  # Use identity matrix at fc-layer, to get 512 neuron output
                                            # Replaces the final layer to be an output layer
        self.model_conv.eval()
        
        # freeze the gradients
        for param in self.model_conv.parameters():
            param.requires_grad = False
        
        self.fc1 = nn.Linear(512, output_dim)  # input should be: out_channels times image size times image size 
        self.fc2 = nn.Linear(output_dim, 1)
        
        # Add dropout
        self.dropout = nn.Dropout(dropout)
    
    
    def forward(self, x):
        x = self.model_conv(x)
       
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        return torch.sigmoid(x)
    
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        images, labels = batch
        p_labels = self(images).reshape(-1)  # reshape to get shape (50) instead of (50, 1)
        loss = F.binary_cross_entropy(p_labels, labels)
        acc = (labels == torch.round(p_labels)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss    
    
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        p_labels = self(images).reshape(-1)
        loss = F.binary_cross_entropy(p_labels, labels)
        acc = (labels == torch.round(p_labels)).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
