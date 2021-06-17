from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core import Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import Model

def main():

    parser = argparse.ArgumentParser(description='Training arguments', usage="python deploy_model.py <command>")
    parser.add_argument('-compute')
    args = parser.parse_args(sys.argv[1:])


    if args.compute:
        # requires a config.json file in the same directory
        ws = Workspace.from_config()

        # getting the model
        model = ws.models['tumor_identification_model']

        # Add the dependencies for our model (AzureML defaults is already included)
        myenv = CondaDependencies()
        myenv.add_conda_package('torch', 'torchvision', 'pytorch-lightning')

        # Save the environment config as a .yml file
        env_file = os.path.join('./',"tumor_env.yml")
        with open(env_file,"w") as f:
            f.write(myenv.serialize_to_string())
        print("Saved dependency info in", env_file)

        # Configure the scoring environment
        inference_config = InferenceConfig(runtime= "python",
                                        entry_script='predict_script.py',
                                        conda_file=env_file)

        deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

        service_name = 'tumor-identification-service'

        service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)

        service.wait_for_deployment(True)
        print(service.state)
    else:
        raise ValueError('Please select a compute to run on.')


if __name__ == "__main__":
    main()