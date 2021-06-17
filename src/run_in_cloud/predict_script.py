import json
import joblib
import numpy as np
from azureml.core.model import Model
import pytorch_lightning as pl

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('../models/model.py')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])

    # Create a trainer
    trainer = pl.Trainer(max_epochs=3)

    # Get predictions from the trainer, TODO: data needs to be a dataloader
    predictions = trainer.predict(model, data) # returns a list of dictionaries, one for each provided dataloader containing their respective predictions
    predicted_classes = []
    classnames = ['yes', 'no']

    # TODO: needs reviewing
    # for prediction,  in predictions:
    #     predicted_classes.append(classnames[prediction])
    # Return the predictions as JSON
    return json.dumps(predicted_classes)