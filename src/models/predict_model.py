import torch
from torchvision import transforms
import pytorch_lightning as pl
import argparse, sys
import numpy as np

from torch.cuda import is_available
from src.models.model import TumorClassifier
from PIL import Image
import kornia as K


def predict():
    filename = 'src/models/checkpoints/tumorclassfier_epoch=194_val_acc=0.92.ckpt'
    model = TumorClassifier.load_from_checkpoint(filename, output_dim=115, dropout=0.2852319202280339)
    model.eval()
    
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            #transforms.Normalize((0.5,), (0.5,))
        ]
    )

    #random_image = 'brain_tumor_dataset/intermediate/yes/Y2_rotate.jpg'
    random_image = 'brain_tumor_dataset/intermediate/no/3 no_blur.jpg'
    
    img = Image.open(random_image).convert('RGB')
    processed_img = preprocess(img) # .reshape((1, 3, 224, 224))
    processed_img = K.image_to_tensor(np.asarray(processed_img), keepdim=False).float()
    p_labels = model(processed_img.reshape((1, 3, 224, 224)))
    print(p_labels)


if __name__ == '__main__':
    predict()

