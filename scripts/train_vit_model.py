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

from my_vit import ViT

from dataloaders import get_CelebA_loader, get_PetExpression_loader, get_AltPet_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

BATCH_SIZE = 32
BATCH_SIZE_TEST = 32
NC = 3 # 1 # 
IMG_H = IMG_W = 64
EPOCHS = 2500
DATA = 'PetExpression'

#set up working data location
working_root = './'

if DATA == 'CelebA':
    data_root = './data/celeba/50k/'
    train_loader = get_CelebA_loader(data_root + "train.txt", BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment = True, shuffle = True, num_workers = 1, pin_memory = True) # Shape (C, H, W)
    eval_loader = get_CelebA_loader(data_root + "eval.txt", BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment = False, shuffle = True, num_workers = 1, pin_memory = True)
    test_loader = get_CelebA_loader(data_root + "test.txt", BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment = False, shuffle = True, num_workers = 1, pin_memory = True)
elif DATA == 'PetExpression':
    data_root = "./data/PetExpression/Master Folder/"
    train_loader = get_PetExpression_loader(data_root+'train/', batch_size=BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment=True, shuffle=True, num_workers=1, pin_memory=True)
    eval_loader = get_PetExpression_loader(data_root+'test/', batch_size=BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment=False, shuffle=False, num_workers=1, pin_memory=True)
elif DATA == 'Cifar':
    data_root = "./data/cifar/cifar10-64/"
    train_loader = get_PetExpression_loader(data_root+'train/', batch_size=BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment=True, shuffle=True, num_workers=1, pin_memory=True)
    eval_loader = get_PetExpression_loader(data_root+'test/', batch_size=BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment=False, shuffle=False, num_workers=1, pin_memory=True)
elif DATA == 'AltPet':
    data_root = './data/AltPets/'
    train_loader = get_AltPet_loader(data_root + "train_filenames.txt", BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment = True, shuffle = True, num_workers = 1, pin_memory = True) # Shape (C, H, W)
    eval_loader = get_AltPet_loader(data_root + "valid_filenames.txt", BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment = False, shuffle = True, num_workers = 1, pin_memory = True)
    test_loader = get_AltPet_loader(data_root + "test_filenames.txt", BATCH_SIZE_TEST, shape = (NC, IMG_H, IMG_W), augment = False, shuffle = True, num_workers = 1, pin_memory = True)
    matrix_loader = get_AltPet_loader(data_root + "matrix_filenames.txt", BATCH_SIZE_TEST, shape = (NC, IMG_H, IMG_W), augment = False, shuffle = True, num_workers = 1, pin_memory = True)
else:
    print('Data source not recognized')
    assert False

device = "cuda" if torch.cuda.is_available() else "cpu"

main_loader = train_loader
eval_loader = eval_loader


config = BaseTrainerConfig(
    output_dir='./checkpoints/my_plain_vae_model',
    learning_rate=0.0005,#5e-4,
    #scheduler_cls='CosineAnnealingLR',
    #scheduler_params={'T_max': EPOCHS, 'eta_min':1e-8},
    #scheduler_cls='CosineAnnealingWarmRestarts',
    #scheduler_params={'T_0': 50, 'eta_min':1e-6},
    scheduler_cls='CyclicLR',
    scheduler_params={'base_lr':1e-5, 'max_lr':0.001, 'mode':'triangular2','cycle_momentum':False},
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE_TEST,
    num_epochs=EPOCHS,
    no_cuda=False,
    steps_saving=125
)

model_config = VAEConfig(
    input_dim=(NC, IMG_H, IMG_W),
    latent_dim=128,
    reconstruction_loss="mse"
)

class Encoder(BaseEncoder):
    """ Usual encoder followed by an exponential map """
    def __init__(self, model_config, prior_iso=False):
        super(Encoder, self).__init__()
        self.nc = NC
        self.ndf = 128
        self.dropout = nn.Dropout(p=0.5)
        # [b, 784] => [b, 20]
        # u: [b, 10]
        # sigma: [b, 10]
        self.vit = ViT(
                        image_size = 64,
                        patch_size = 16,
                        num_classes = 128,
                        dim = 1024,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 512,
                        dropout = 0.1,
                        emb_dropout = 0.1
        )
        self.fc1 = nn.Linear(128, model_config.latent_dim)
        self.fc2 = nn.Linear(128, model_config.latent_dim)

        self.activation = nn.ReLU() #nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.vit(x)
        mu = self.activation(self.fc1(h))
    
        return ModelOutput(
            embedding=mu,
            log_covariance=torch.log(F.softplus(self.fc2(h)) + 1e-5), # expects log_covariance
        )

### Define paper decoder network
class Decoder(BaseDecoder):
    """ First layer is a Hypergyroplane followed by usual decoder """
    def __init__(self, model_config):
        super(Decoder, self).__init__()
        self.input_dim = model_config.input_dim
        self.nc = NC
        self.ngf = 128
        self.linear_up = nn.Sequential(
            nn.Linear(model_config.latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 1000),
            nn.BatchNorm1d(1000),
            nn.GELU(),
            nn.Linear(1000, self.input_dim[0] * self.input_dim[1] * self.input_dim[2]),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.linear_up(z)
        out = out.reshape((z.shape[0],self.input_dim[0], self.input_dim[1], self.input_dim[2]))  # reshape data
        
        return ModelOutput(
            reconstruction=out
        )

# Build the model
model = VAE(
    model_config=model_config,
    encoder=Encoder(model_config),
    decoder=Decoder(model_config),
)

print(count_parameters(model))

# Build the Pipeline
pipeline = TrainingPipeline(
    training_config=config,
    model=model
)

# Launch the Pipeline
pipeline(
    train_data=main_loader,
    eval_data=eval_loader
)

last_training = sorted(os.listdir('./checkpoints/my_plain_vae_model'))[-1]
trained_model = AutoModel.load_from_folder(os.path.join('./checkpoints/my_plain_vae_model', last_training, 'final_model')).to(device)

# Generate Data
# create normal sampler
normal_sampler = NormalSampler(
    model=trained_model
)

# sample
gen_data = normal_sampler.sample(
    num_samples=16
)

pu = PlottingUtils(trained_model, viz_size=4, data = next(iter(eval_loader))['data'].to(device))
pu.generated_from_samples(gen_data)
pu.original("height")
pu.original('img')
pu.reconstructed("height")
pu.reconstructed('img')
pu.show()
