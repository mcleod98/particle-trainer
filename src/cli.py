from train import train_net
from utils import get_session
from db_models import Img
from nn_models import SimpleNet
from dataset import SimulatedParticleDataset
import click, os, yaml, time

default_config = os.path.join('/opt/notebooks', 'default_config.yaml')
default_hyper = os.path.join('/opt/notebooks', 'default_hyper.yaml')

@click.group()
def cli():
    pass

@cli.command()
@click.argument('config_filepath', type=click.Path(exists=True), required=False)
@click.argument('hyper_filepath', type=click.Path(exists=True), required=False)
def train(config_filepath, hyper_filepath):
    '''
    Entrypoint for training neural network

    Args: config_filepath: Yaml file with configuration parameters for I/O paths and database configuration
          hyper_filepath: Yaml file with hyperparameters for training

    Output is saved to directories specified in config
    '''

    if not config_filepath:
        config_filepath = default_config
    if not hyper_filepath:
        hyper_filepath = default_hyper
    with open(config_filepath, "r") as f:
        conf = yaml.full_load(f)
    with open(hyper_filepath, "r") as g:
        hyper = yaml.full_load(g)
    session = get_session(conf)

    imgs = session.query(Img).all()

    v = [im for j, im in enumerate(imgs) if j % 5 == 0]
    t = [im for j, im in enumerate(imgs) if j % 5 != 0]
    print(f'Splitting {len(imgs)} images into {len(t)} training samples and {len(v)} validation samples')

    trainset = SimulatedParticleDataset(img_objs = t, img_dir =conf['img_dir'])
    valset = SimulatedParticleDataset(img_objs = v, img_dir =conf['img_dir'])

    model = SimpleNet()

    if conf.get('model_load', None):
        model.load_weights(conf['model_load'])
        
    train_net(model, trainset, valset, hyper, conf)

if __name__ == '__main__':
    cli()




