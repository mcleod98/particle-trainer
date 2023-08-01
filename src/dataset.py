import torch
import os

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SimulatedParticleDataset(Dataset):
    """
    Simulated particles dataset. Takes a list of 'Img' objects
    Loads binary image from filepath at obj.img_filepath 
    Generates normalized targets from obj.particles
    """

    def __init__(self, img_objs, img_dir=None,):
        """
        Arguments:
            img_objs (list): A list of image objects
            img_dir (string): Path to directory that contains the image files.
                If None, uses directory from img_objs
        """
        self.objs = img_objs
        self.img_dir = img_dir

    def __len__(self):
        return len(self.objs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_full_path = self.objs[idx].img_filepath
        if self.img_dir:
            _, filename = os.path.split(img_full_path)
            img_full_path = os.path.join(self.img_dir, filename)
            
        with Image.open(img_full_path) as img:
            img.load()
            
        f = transforms.PILToTensor()(img).to(torch.float)
        
        p = self.transform_targets(self.objs[idx].particles)
        sample = {'image': f, 'particles': p}


        return sample
    
    def transform_targets(self, batch):
        p = torch.zeros(20, 5)
        xscale = 250
        yscale = 250
        amin = 5
        bmin = 5
        bmax = 60
        amax = 60
        particles = [torch.tensor([1, p.x/xscale, p.y/yscale, (p.a - amin) / (amax-amin), (p.b - bmin) / (bmax - bmin)], dtype=torch.float) for p in batch]
        if len(particles) > 0:
            _p = torch.stack(particles)
            p[:len(_p), :] = _p
        return p