import torch
import argparse, sys
import os

from torchvision.models.resnet import ResNet
from src.models.model import TumorClassifier
from torchvision.models import resnet18

def model_to_torchserve():
    """
    Takes model and makes it compatible with Torchserve.
    Arguments can be given in the command line with the --model_path flag.
    """

    parser = argparse.ArgumentParser(description='Path to model that has to be formatted to fit Torchserve')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--output_dim', default=64, type=int)
    parser.add_argument('--dropout', default=0.25, type=float)
    args = parser.parse_args(sys.argv[1:])
    print(args)
    
    if args.model_path is None:
        raise ValueError('Please give a path to the model.')
    else:
        if os.path.isfile(args.model_path):
            os.makedirs('./models/torchserve_models', exist_ok=True)  # Create torchserve_models folder if it doesn't exist
            
            model = TumorClassifier.load_from_checkpoint(args.model_path, output_dim=args.output_dim, dropout=args.dropout)
            script_model = torch.jit.script(model)
            script_model.save(f'models/torchserve_models/deployable_model.pt')
            
        else:
            raise ValueError(f'Could not find file at given path: {args.model_path}')
    
if __name__ == '__main__':
    model_to_torchserve()
