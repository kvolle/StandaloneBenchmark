
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from pythae.models import PoincareVAE, PoincareVAEConfig, AutoModel
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline

import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pythae.samplers import PoincareDiskSampler, NormalSampler
from pythae.models.pvae.pvae_utils import PoincareBall

import os
from glob import glob

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

from plotting_utils import PlottingUtils

import cv2
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

from transformers import get_cosine_schedule_with_warmup

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_np(a):
    return a.clone().detach().cpu().numpy()

config = {'lr':1e-3,
          'wd':1e-2,
          'bs':256,
          'channels':3,
          'img_size':128,
          'epochs':100,
          'seed':1000}

IMAGENET_ROOT = '../StandaloneBenchmark/data/TinyImageNet/'
CIFAR_ROOT = '../StandaloneBenchmark/data/cifar/cifar10-64/'
ALTPET_ROOT = '../StandaloneBenchmark/data/AltPets/'
CELEB_ROOT = '../StandaloneBenchmark/data/celeba/50k/'
CARS_ROOT = '../../Misc/Datasets/StanfordCars/'
MARINE_ROOT = '../../Misc/Datasets/VisualMarineAnimalTracking/jpgs-001/jpgs/'
ITALIAN_ANIMALS_ROOT = '../../Misc/Datasets/Animals-10/raw-img/'
ANIMAL_FACE_ROOT = '../StandaloneBenchmark/data/afhq/'
IMAGENET_R_ROOT = '../../Misc/Datasets/imagenet-r/'
VANGOGH_ROOT = '../../Misc/Datasets/VanGogh/VincentVanGogh/'

def delocalize(filenames):
    f = open(filenames, 'r')
    names = f.read().splitlines()
    f.close()
    for i in range(len(names)):
        rel = names[i].split('./')
        names[i] = ALTPET_ROOT +rel[-1]
    return np.asarray(names)

def celeb_names(split):
    f = open(CELEB_ROOT+split+'.txt')
    names = f.read().splitlines()
    f.close()
    for i in range(len(names)):
        names[i] = CELEB_ROOT + names[i] + '.jpg'
    return np.asarray(names)


train_paths = delocalize(ALTPET_ROOT + 'train_filenames.txt')
valid_paths = delocalize(ALTPET_ROOT + 'valid_filenames.txt')


IMG_MEAN = [0.485, 0.456, 0.406] # [0.0, 0.0, 0.0] # 
IMG_STD = [0.229, 0.224, 0.225] # [1.0, 1.0, 1.0] # 

normalization = {'mean':IMG_MEAN, 'std':IMG_STD}

"""
def get_train_transforms():
    return A.Compose(
        [
            A.Resize(config['img_size'],config['img_size'],always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ])
"""
def get_train_transforms():
    return A.Compose(
        [
            A.Resize(2*config['img_size'], 2*config['img_size'], always_apply=True),
            A.SafeRotate(always_apply=False, p=0.3, limit=(-30, 30), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
            A.CenterCrop(int(np.floor(1.3*config['img_size'])), int(np.floor(1.3*config['img_size'])), always_apply=True),
            A.CoarseDropout(always_apply=False, p=1.0, max_holes=8, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=(124, 116, 104), mask_fill_value=None),
            A.Resize(config['img_size'],config['img_size'],always_apply=True),
            A.Normalize(),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(p=1.0)
        ])

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(config['img_size'],config['img_size'],always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ])
   
class ImageNetDataset(Dataset):
    def __init__(self,paths,augmentations):
        self.paths = paths
        self.augmentations = augmentations
    
    def __getitem__(self,idx):
        path = self.paths[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        
        return {'data':image}
    
    def __len__(self):
        return len(self.paths)
        
test_dataset = ImageNetDataset(valid_paths,augmentations=get_train_transforms())
test_dl = DataLoader(test_dataset,batch_size=16,shuffle=False,num_workers=4)

dataiter = iter(test_dl)
sample = next(dataiter)


device = "cuda" if torch.cuda.is_available() else "cpu"

## train
train_dataset = ImageNetDataset(train_paths,augmentations=get_train_transforms())
main_loader = DataLoader(train_dataset,batch_size=config['bs'],shuffle=True,num_workers=4)

#valid
valid_dataset = ImageNetDataset(valid_paths,augmentations=get_valid_transforms())
eval_loader = DataLoader(valid_dataset,batch_size=config['bs'],shuffle=False,num_workers=4)

training_config = BaseTrainerConfig(
    output_dir='./checkpoints/my_plain_vae_model',
    learning_rate=config['lr'],#5e-4,
    scheduler_cls='CosineAnnealingLR',
    scheduler_params={'T_max': config['epochs']*len(main_loader), 'eta_min':1e-8},
    optimizer_cls='AdamW',
    optimizer_params={'weight_decay':config['wd']},
    per_device_train_batch_size=config['bs'],
    per_device_eval_batch_size=config['bs'],
    num_epochs=config['epochs'],
    no_cuda=False,
    steps_saving=125
)

model_config = PoincareVAEConfig(
    input_dim=(config['channels'], config['img_size'], config['img_size']),
    latent_dim=2,#56,
    reconstruction_loss="mse",
    prior_distribution='riemannian_normal',
    posterior_distribution='wrapped_normal',
    curvature=0.7
)
# Let's define some custom Encoder/Decoder to stick to the paper proposal
### Define paper encoder network
class Encoder(BaseEncoder):
    """ Usual encoder followed by an exponential map """
    def __init__(self, model_config, prior_iso=False):
        super(Encoder, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.shape = 32 # TODO examine this
        self.manifold = PoincareBall(dim=model_config.latent_dim, c=model_config.curvature)

        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*(self.shape-1)**2,2*self.latent_dim)
        self.activation = nn.ReLU()
        self.scale = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        #x = self.bn0(x)
        h1 = self.activation(self.conv1(x))
        h2 = self.activation(self.conv2(h1))
        h3 = self.activation(self.flatten(h2))
        h4 = self.fc1(h3)
        mu, logvar = torch.split(h4, self.latent_dim, dim=1)
        mu = self.manifold.expmap0(mu)
        logvar = torch.log(F.softplus(logvar) + 1e-6)
        
        return ModelOutput(
            embedding=mu,
            log_covariance=logvar,  #torch.log(F.softplus(logvar) + 1e-5), # expects log_covariance
            log_concentration=logvar
        )

### Define paper decoder network
class Decoder(BaseDecoder):
    """ First layer is a Hypergyroplane followed by usual decoder """
    def __init__(self, model_config):
        super(Decoder, self).__init__()
        self.input_dim = model_config.input_dim
        self.latent_dim = model_config.latent_dim
        self.nc = config['channels']
        self.shape = 32

        self.fc2 = nn.Linear(self.latent_dim,(self.shape**2) *32)
        self.conv3 = nn.ConvTranspose2d(32,64,kernel_size=2,stride=2)
        self.conv4 = nn.ConvTranspose2d(64,32,kernel_size=2,stride=2)
        self.conv5 = nn.ConvTranspose2d(32,3,kernel_size=1,stride=1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.activation = self.relu # self.leakyrelu # 
    def forward(self, z):
        #out = self.dec(z).reshape((z.shape[0],) + self.input_dim)  # reshape data
        h1 = self.activation(self.fc2(z))
        h1 = h1.view(-1, self.shape, self.shape, self.shape)
        h2 = self.activation(self.conv3(h1))
        h3 = self.activation(self.conv4(h2))
        h4 = self.conv5(h3)
        return ModelOutput(
            reconstruction=h4
        )
    
# Build the model
model = PoincareVAE(
    model_config=model_config,
    encoder=Encoder(model_config),
    decoder=Decoder(model_config),
)

print(count_parameters(model))

# Build the Pipeline
pipeline = TrainingPipeline(
    training_config=training_config,
    model=model
)

# Launch the Pipeline
pipeline(
    train_data=main_loader,
    eval_data=eval_loader
)

last_training = sorted(os.listdir('./checkpoints/my_plain_vae_model'))[-1]
print(last_training)
trained_model = AutoModel.load_from_folder(os.path.join('./checkpoints/my_plain_vae_model', last_training, 'final_model')).to(device)

# Generate Data
# create normal sampler
pvae_sampler = PoincareDiskSampler(
    model=trained_model
)

# sample
gen_data = pvae_sampler.sample(
    num_samples=16
)

pu = PlottingUtils(trained_model, viz_size=4, data = next(iter(eval_loader))['data'].to(device), normalization_used=normalization)
pu.generated_from_samples(gen_data)
pu.original("height")
pu.original('img')
pu.reconstructed("height")
pu.reconstructed('img')
pu.show()
