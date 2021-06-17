# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
# import kaggle
import os
import sys
import argparse

from os import path
from PIL import Image
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split

@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # TODO: create method how to get raw data
    # TODO: make check to see if the folder structure is there
    
    # TODO: use kornia to do image augmentation

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5,), (0.5,))
        ]
    )

    data = []
    targets = []

    for i, dir in enumerate(os.listdir('brain_tumor_dataset/raw/')):
        for filename in os.listdir(f'brain_tumor_dataset/raw/{dir}'):
            g_image = Image.open(f'brain_tumor_dataset/raw/{dir}/{filename}').convert('L')
            processed = preprocess(g_image).detach().numpy()

            data.append(processed)
            targets.append(i)
    
    data    = torch.tensor(data)
    targets = torch.tensor(targets)

    X_train, X_test, y_train, y_test = train_test_split(data, targets, 
                                                        test_size=0.2, 
                                                        stratify=targets, 
                                                        random_state=42)

    train_dict = {'data': X_train, 'target': y_train}
    test_dict  = {'data': X_test, 'target': y_test}
    
    torch.save(train_dict, 'brain_tumor_dataset/processed/train.pt')
    torch.save(test_dict, 'brain_tumor_dataset/processed/test.pt')


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
            print("Directories created successfully!")
        except OSError as error:
            print("Directories can not be created")
    else:
        main()
