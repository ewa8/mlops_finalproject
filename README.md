mlops_finalproject
==============================

# MLOps Project Description 
Ewa - s203262, Stefan - s173991, Magnus - s164415, Selma - s163740 

*Date: 10.06.21*

## Overall goal

The goal of the project is to use an open-source framework to apply computer vision algorithmson brain tumor data and use a pre-trained model to classify the images.

## Framework

We plan to use the Kornia framework for the project. We will use the data augmentation feature toincrease the sample size of our data. Additionally, we consider using the functionality of sharpeningthe  images  to  improve  the  quality  of  existing  data  with  the  intention  of  making  features  moredistinct.

## Data 

The data set used for the project is called [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection?fbclid=IwAR1E8c2ZIR4g4ePLUku6PWjESgeHClsqTXmWqPZSA4ut2DLleNBH6GbiwYw) and is found on Kaggle.  It is a binary classification problem where we need to classify whether a braincontains a tumor or not.  It consists of 253 brain MRI scan images, where 98 of the brain scansbelongs to the class ”No” and the remaining 155 images belongs to the class ”Yes”.  The imagesvaries in size with some of the largest ones being 1024×1024 and 1920×1080 compared to thesmallest images being 201×251 and 300×168.

## Models

We are going to use a pre-trained network (transfer learning) as backbone for our convolutionalneural network.  This is to further increase efficiency of our training as data sample size is alreadysparse.  Furthermore, the pre-trained network will be chosen from the already supported modelsin pytorch, example:  ResNet.  Most of pytorch’s pre-trained models takes RGB images of a certainsize eg.  224×224 for ResNet, we therefore need to keep this in mind when choosing pre-trainedmodels.

![](/reports/figures/data-example.png)
 Figure a shows a MRI scan image of a brain with a tumor, belonging to category ”Yes”.  b: A MRI scan image of a healthy brain, belonging to category ”No”.


Repo for the final project for MLOps course at DTU

### TODO

- [x] Create a git repository
- [x] Make sure that all team members have write access to the github repository
- [x] Create a dedicated environment for you project to keep track of your packages
- [x] Create the initial file structure using cookiecutter
- [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and 
- [ ] Use Kornia
- [ ] Add a model file and a training script and get that running
- [ ] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code
- [ ] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
- [ ] Write unit tests for some part of the codebase and calculate the 
- [ ] Get some continues integration running on the github repository
- [ ] use either tensorboard or wandb to log training progress and other important metrics/artifacts in your code
- [ ] remember to comply with good coding practices while doing the project
- [ ] Setup and used Azure to train your model
- [ ] Played around with distributed data loading
- [ ] (not curriculum) Reformated your code in the pytorch lightning format
- [ ] Deployed your model using Azure
- [ ] Checked how robust your model is towards data drifting
- [ ] Deployed your model locally using TorchServe
- [ ] Used Optuna to run hyperparameter optimization on your model
- [ ] Wrote one or multiple configurations files for your experiments
- [ ] Used Hydra to load the configurations and manage your hyperparameters


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
# mlops_finalproject

