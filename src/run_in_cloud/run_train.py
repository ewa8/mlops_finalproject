import azureml.core
from azureml.core import Workspace
from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core import Model
from azureml.core.conda_dependencies import CondaDependencies
from azureml.widgets import RunDetails
import argparse


def main():
    parser = argparse.ArgumentParser(description='Training arguments', usage="python run_train.py <command>")
    parser.add_argument('-compute')
    args = parser.parse_args(sys.argv[1:])

    if args.compute:
        #requires a config.json file in the same directory
        ws = Workspace.from_config()

        cluster = ws.compute_targets[args.compute]

        # Create a Python environment for the experiment
        pytorch_env = Environment("pytorch-env")

        # Ensure the required packages are installed (we need pip, scikit-learn and Azure ML defaults)
        packages = CondaDependencies.create(conda_packages=['pip'],
                                            pip_packages=['azureml-defaults', 'torch', 'torchvision', 'pytorch-lightning', 'wandb', 'azureml-mlflow', 'mlflow'])
        pytorch_env.python.conda_dependencies = packages

        # Create a script config
        training_folder = '../models/'
        script_config = ScriptRunConfig(source_directory=training_folder,
                                        script='train_model.py',
                                        environment=pytorch_env, 
                                        compute_target=cluster) 

        # submit the experiment
        experiment_name = 'training_experiment'
        experiment = Experiment(workspace=ws, name=experiment_name)
        run = experiment.submit(config=script_config)
        # RunDetails(run).show()
        run.wait_for_completion()

        # Register the model
        run.register_model(model_path='../models/model.pt', model_name='tumor_identification_model',
                   tags={'Training context':'Script'},
                #    properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']}
                   )

    else:
        raise ValueError('Please select a compute to run on.')

if __name__ == "__main__":
    main()