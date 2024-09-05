
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from pythae.models import VAE, VAEConfig, AutoModel
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline

import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pythae.samplers import PoincareDiskSampler, NormalSampler

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

config = {'lr':1e-3,
          'wd':1e-5,
          'bs':256,
          'channels':1,
          'img_size':32,
          'epochs':25,
          'seed':1000}


IMG_MEAN = [0.485, 0.456, 0.406] # [0.0, 0.0, 0.0] # 
IMG_STD = [0.229, 0.224, 0.225] # [1.0, 1.0, 1.0] # 

normalization = {'mean':IMG_MEAN, 'std':IMG_STD}


def get_train_transforms():
    return transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Pad(padding=[2]),
                            transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])

def get_valid_transforms():
    return transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Pad(padding=[2]),
                            transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
   
class VAEDataset(Dataset):
    def __init__(self,torchvision_dataset):
        self.dataset = torchvision_dataset
    
    def __getitem__(self,idx):
        (image, target) = self.dataset.__getitem__(idx)

        return {'data':image, 'key':target}
    
    def __len__(self):
        return self.dataset.__len__()
        
#test_dataset = ImageNetDataset(valid_paths,augmentations=get_train_transforms())
#test_dl = DataLoader(test_dataset,batch_size=16,shuffle=False,num_workers=4)

test_dataset = VAEDataset(datasets.MNIST('./data/', train=False, download=True, transform=get_valid_transforms()))
test_dl = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

dataiter = iter(test_dl)
sample = next(dataiter)

device = "cuda" if torch.cuda.is_available() else "cpu"

## train
#train_dataset = ImageNetDataset(train_paths,augmentations=get_train_transforms())
#main_loader = DataLoader(train_dataset,batch_size=config['bs'],shuffle=True,num_workers=4)
train_dataset = VAEDataset(datasets.MNIST('./data/', train=True, download=True, transform=get_train_transforms()))
main_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)


#valid
#valid_dataset = ImageNetDataset(valid_paths,augmentations=get_valid_transforms())
#eval_loader = DataLoader(valid_dataset,batch_size=config['bs'],shuffle=False,num_workers=4)
eval_loader = test_dl

training_config = BaseTrainerConfig(
    output_dir='./checkpoints/my_ea_vae_model',
    learning_rate=config['lr'],#5e-4,
    #scheduler_cls='CosineAnnealingLR',
    #scheduler_params={'T_max': config['epochs']*len(main_loader), 'eta_min':1e-8},
    ##scheduler_cls='CosineAnnealingWarmRestarts',
    ##scheduler_params={'T_0': 50, 'eta_min':1e-6},
    ##scheduler_cls='CyclicLR',
    ##scheduler_params={'base_lr':1e-5, 'max_lr':0.001, 'mode':'triangular2','cycle_momentum':False},
    optimizer_cls='AdamW',
    optimizer_params={'weight_decay':config['wd']},
    per_device_train_batch_size=config['bs'],
    per_device_eval_batch_size=config['bs'],
    num_epochs=config['epochs'],
    no_cuda=False,
    steps_saving=125
)

model_config = VAEConfig(
    input_dim=(config['channels'], config['img_size'], config['img_size']),
    latent_dim=4,
    reconstruction_loss='mse'#"nll"
)
# Let's define some custom Encoder/Decoder to stick to the paper proposal
### Define paper encoder network
class Encoder(BaseEncoder):
    """ Usual encoder followed by an exponential map """
    def __init__(self, model_config, ea = False, prior_iso=False):
        super(Encoder, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.shape = 5 # TODO examine this
        self.ea = ea # TODO implement this
        self.conv1 = nn.Conv2d(  config['channels'],  64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d( 64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*(self.shape-1)**2, 2*self.latent_dim)
        self.activation = nn.ReLU()
        self.scale = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        #x = self.bn0(x)
        h1 = self.activation(self.conv1(x))
        h2 = self.activation(self.conv2(h1))
        h3 = self.activation(self.conv3(h2))
        h4 = self.activation(self.flatten(h3))
        h5 = self.fc1(h4)
        mu, logvar = torch.split(h5, self.latent_dim, dim=1)
        #print(h.shape for h in [h1, h2, h3, h4, h5])
        #assert False

        
        return ModelOutput(
            embedding=mu,
            log_covariance = torch.log(F.softplus(logvar) + 1e-5), # expects log_covariance # logvar
        )

### Define paper decoder network
class Decoder(BaseDecoder):
    """ First layer is a Hypergyroplane followed by usual decoder """
    def __init__(self, model_config, ea=False):
        super(Decoder, self).__init__()
        self.input_dim = model_config.input_dim
        self.latent_dim = model_config.latent_dim
        self.nc = config['channels']
        self.shape = 4
        self.ea = ea
        self.fc2 = nn.Linear(self.latent_dim,(self.shape**2) *256)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2,stride=2)
        self.conv4 = nn.ConvTranspose2d(128,  64, kernel_size=2,stride=2)
        self.conv5 = nn.ConvTranspose2d( 64,   config['channels'], kernel_size=2,stride=2)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.activation = self.relu # self.leakyrelu # 
    def forward(self, z):
        #out = self.dec(z).reshape((z.shape[0],) + self.input_dim)  # reshape data
        if self.ea:
            z, z_mult = torch.split(z, self.latent_dim, dim=1)
        h1 = self.activation(self.fc2(z))
        #h1 = h1.view(-1, self.shape, self.shape-1, self.shape)
        h1 = h1.view(-1, 256, self.shape, self.shape)
        h2 = self.activation(self.conv3(h1))
        h3 = self.activation(self.conv4(h2))
        if self.ea:
            h4 = self.sigmoid(self.conv5(h3)*z_mult) #TODO This scaling doesn't make sense
        else:
            h4 = self.sigmoid(self.conv5(h3))
        return ModelOutput(
            reconstruction=h4
        )
    
# Build the model
EA_Version= False
model = VAE(
    model_config=model_config,
    encoder=Encoder(model_config, ea = EA_Version),
    decoder=Decoder(model_config, ea = EA_Version),
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

last_training = sorted(os.listdir('./checkpoints/my_ea_vae_model'))[-1]
print(last_training)
trained_model = AutoModel.load_from_folder(os.path.join('./checkpoints/my_ea_vae_model', last_training, 'final_model')).to(device)

# Generate Data
# create normal sampler
normal_sampler = NormalSampler(
    model=trained_model
)

# sample
gen_data = normal_sampler.sample(
    num_samples=16
)

pu = PlottingUtils(trained_model, viz_size=4, data = next(iter(eval_loader))['data'].to(device), normalization_used=normalization)
pu.generated_from_samples(gen_data)
pu.original("height")
pu.original('img')
pu.reconstructed("height")
pu.reconstructed('img')
pu.show()
