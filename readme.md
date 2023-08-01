# Synthetic Particle Training

This is a python program for training a neural network on a tagged image dataset stored in an sql database.

Includes Dataclass model for loading from the database using lists of mapped sqlalchemy objects, and custom loss function for dealing with multiple predictions and varying number of targets. 

## Getting Started

1. Pull this repository
2. Install docker and build container from provided dockerfile with `docker build . --tag 'particles_nn'` Note that the container name has to match the image name in our docker-compose.yml file
3. Configure .yaml files at `data/default_config.yaml` and `data/default_hyper.yaml` to configure input and output directories/filepaths and training hyperparameters.
4. Move data files (image directory and sqlite database) to the data subdirectory, this volume is copied into our docker container so these files will be available to us for training
5. Run `docker compose up` to begin a jupyter notebook server in our container. Or run `docker compose run python -u "/opt/src/cli.py" train` to use the command line entrypoint


## Config.yaml

```
    model_load: Filepath to load model from
    model_save: Filepath to save final trained model
    img_dir: Path to directory where image files are located (if it's not the same directory as in sql database)
    sqlite_path: Path to sqlite database
    db_type: Recognizes SQLITE or POSTGRES, configures sqlalchemy engine object to connect to desired database
    checkpoint_load: Filepath to checkpoint to resume model training from
    checkpoint_save: Filepath to output training checkpoint
    
```

## Hyper.yaml

```
    learning_rate: Learning rate parameter to be passed to the optimizer
    batch_size: Batch size parameter for DataLoader
    epochs: Number of training epochs
    loss_weights: a list of 5 weights corresponding to relative loss weightings for different target columns:
        [ probability, x, y, a, b ]
```
