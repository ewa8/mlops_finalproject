import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from azureml.core import Run
from pytorch_lightning.loggers import MLFlowLogger

from src.data.data_module import DataModule

import argparse
import sys

from src.models.model import TumorClassifier

def train():
    print('Training model')
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--use_wandb', default=False)
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[1:])
    print(args)

    run =  Run.get_context()
    mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri()

    mlf_logger = MLFlowLogger(experiment_name=run.experiment.name, tracking_uri=mlflow_url)
    mlf_logger._run_id = run.id

    if args.use_wandb:  # Use wandb logger
        print('Using wandb...')
        kwargs = {'entity': 'sems'}
        chosen_logger = WandbLogger(project='final_project', **kwargs)
    else:  # Use default logger
        chosen_logger = True
    
    model = TumorClassifier()
    data = DataModule(data_dir='./brain_tumor_dataset/processed', batch_size=24)
    
    trainer = pl.Trainer(logger=chosen_logger, max_epochs=10, log_every_n_steps=2)
    trainer.fit(model, datamodule=data)
    

# def evaluate():
#     print("Evaluating until hitting the ceiling")
#     parser = argparse.ArgumentParser(description='Training arguments')
#     parser.add_argument('--load_model_from', default="checkpoint.pth")
#     # add any additional argument that you want
#     args = parser.parse_args(sys.argv[2:])
#     print(args)
    
#     # TODO: Implement evaluation logic here
#     if args.load_model_from:
#         checkpoint = torch.load('ResNet18.pth')
#     _, test_set = mnist()
#     model = Classifier()
#     model.load_state_dict(checkpoint)

#     # TODO load the test data
#     #testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

#     running_accuracy = []
#     with torch.no_grad():
#         # set model to evaluation mode
#         model.eval()

#         # validation pass here
#         for images, labels in testloader:

#             ps = torch.exp(model(images))

#             top_p, top_class = ps.topk(1, dim=1)

#             equals = top_class == labels.view(top_class.shape)

#             accuracy = torch.mean(equals.type(torch.FloatTensor))
#             running_accuracy.append(accuracy)


#     print(f'Accuracy: {np.mean(running_accuracy)*100}%')


if __name__ == '__main__':
    train()

