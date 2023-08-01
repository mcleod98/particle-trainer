import torch
from torch.nn.utils.rnn import pad_sequence
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def pad_collate(batch):
    '''
    Uses torch.pad_sequence to pad each tensor in batch to the same size, then stack them

    Pass this as collate_fn to our data loader because all samples in a batch must be padded to the same size
    '''
    img = torch.stack([d['image'] for d in batch])
    particles = [d['particles'] for d in batch]
    particle_pad = pad_sequence(particles, batch_first=True, padding_value=0)

    return {'img': img, 'particles': particle_pad,}

def save_checkpoint(checkpoint_path, model, optim, epoch):
    'Saves model, optimizer and epoch to checkpoint_path'
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
        }
    torch.save(checkpoint, checkpoint_path)
    
def resume_checkpoint(checkpoint_path, model, optim):
    'Loads model, optimizer and epoch from checkpoint_path'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    return model, optim, checkpoint['epoch']

def get_session(config):
    """Provides a connection to the database through sqlalchemy session"""
    db_path = get_db_string(config)
    engine= create_engine(db_path, echo=False)
    S = sessionmaker(engine, expire_on_commit=True)
    return S()

def get_db_string(config):
    '''
    Returns connection string for sqlalchemy engine depending on config
    '''
    if config['db_type'] == 'SQLITE':
        return(f'sqlite:///{config["sqlite_path"]}')
    elif config['db_type'] == 'POSTGRES':
        return(f"postgresql://{config['postgres_user']}:{config['postgres_password']}@{config['postgres_host']}:{config['postgres_port']}/{config['postgres_db']}")



