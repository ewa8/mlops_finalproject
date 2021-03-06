# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
# import kaggle
import os
import sys
import argparse
import pytest
from torch import nn

import random

from os import path
from PIL import Image
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split

# Kornia
import kornia as K
import numpy as np
import matplotlib.pyplot as plt # can be removed later
import cv2

@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())

def findFileExtension(filename):

    file_split = filename.split(".")

    extension = file_split[-1]
    filename = file_split[0]
  
    return filename, extension

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    assert True
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # TODO: create method how to get raw data
    # TODO: make check to see if the folder structure is there
    
    # TODO: use kornia to do image augmentation

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            #transforms.ToTensor(),
            #transforms.Normalize((0.5,), (0.5,))
        ]
    )

    data = []
    targets = []
    path = 'brain_tumor_dataset/intermediate/'

    for i, dir in enumerate(os.listdir('brain_tumor_dataset/raw/')):
        print(dir)
        if (dir == '.DS_Store'):
            continue
        for filename in os.listdir(f'brain_tumor_dataset/raw/{dir}'):
            g_image = Image.open(f'brain_tumor_dataset/raw/{dir}/{filename}').convert('RGB')

            processed_img = preprocess(g_image)
            processed: torch.tensor = K.image_to_tensor(np.asarray(processed_img), keepdim=False).float() # Needs to ne numpy array with dim BxCxWxH

            # Data augumentation
            ## Gaussian bluring
            gauss = K.filters.GaussianBlur2d((5, 5), (4, 4))
            blur = gauss(processed.float())

            # Rotation
            # rotate between 45?? and 225??
            alpha: float = random.uniform(45.0, 225.0)  # in degrees
            angle: torch.tensor = torch.ones(1) * alpha

            # define the rotation center
            center: torch.tensor = torch.ones(1, 2)
            center[..., 0] = processed.shape[3] / 2  # x
            center[..., 1] = processed.shape[2] / 2  # y

            # define the scale factor
            scale: torch.tensor = torch.ones((1,2))

            M: torch.tensor = K.get_rotation_matrix2d(center, angle, scale)

            _, _, h, w = processed.shape
            rotate: torch.tensor = K.warp_affine(processed.float(), M, dsize=(h, w))

            # Make tensor to numpy before adding to lists
            # Add original image and argumented imges
            data.append(processed.reshape((3, 224, 224)).detach().numpy())
            data.append(blur.reshape((3, 224, 224)).detach().numpy())
            data.append(rotate.reshape((3, 224, 224)).detach().numpy())

            targets.append(i)
            targets.append(i)
            targets.append(i)

            img_original: np.ndarray = K.tensor_to_image(processed.byte())
            img_blur: np.ndarray = K.tensor_to_image(blur.byte())
            img_rotate: np.ndarray = K.tensor_to_image(rotate.byte()[0])

            # Save to intermediate folder
            cv2.imwrite(os.path.join(path+dir , filename), img_original)

            # TODO use the def findFileExtension
            #filename, extension = findFileExtension(str(filename)) 

            # split the name and extension of the file
            file_split = filename.split(".")

            extension = file_split[-1]
            filename = file_split[0]     

            cv2.imwrite(os.path.join(path+dir , filename+'_blur'+'.'+extension), img_blur)
            cv2.imwrite(os.path.join(path+dir , filename+'_rotate'+'.'+extension), img_rotate)

    
    data    = torch.tensor(data)
    targets = torch.tensor(targets).float()

    X_train, X_test, y_train, y_test = train_test_split(data, targets, 
                                                        test_size=0.2, 
                                                        stratify=targets, 
                                                        random_state=42)

    train_dict = {'data': X_train, 'target': y_train}
    test_dict  = {'data': X_test, 'target': y_test}
    
    torch.save(train_dict, 'brain_tumor_dataset/processed/train.pt')
    torch.save(test_dict, 'brain_tumor_dataset/processed/test.pt')

    ### Just for visulization - can be removed

    # Create the plot
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.ravel()

    axs[0].axis('off')
    axs[0].set_title('image source')
    axs[0].imshow(img_original)

    axs[1].axis('off')
    axs[1].set_title('image blurred')
    axs[1].imshow(img_blur)

    axs[2].axis('off')
    axs[2].set_title('image rotated')
    axs[2].imshow(img_rotate)

    plt.show()
    ###  

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser()
    parser.add_argument('--init', action='store_true')
    args = parser.parse_args()
    init_value = args.init
    
    if init_value:
        try:
            if not (path.exists('brain_tumor_dataset/raw/')):
                os.makedirs('brain_tumor_dataset/raw/')
            if not (path.exists('brain_tumor_dataset/processed')):
                os.makedirs('brain_tumor_dataset/processed')
            if not (path.exists('brain_tumor_dataset/intermediate')):
                os.makedirs('brain_tumor_dataset/intermediate')
                os.makedirs('brain_tumor_dataset/intermediate/yes')
                os.makedirs('brain_tumor_dataset/intermediate/no')
            print("Directories created successfully!")
        except OSError as error:
            print("Directories can not be created")
    else:
        main()