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

from pythae.models.nn.benchmarks.celeba import Decoder_ResNet_AE_CELEBA as Decoder_AE
from pythae.models.nn.benchmarks.celeba import Decoder_ResNet_VQVAE_CELEBA as Decoder_VQVAE
from pythae.models.nn.benchmarks.celeba import Encoder_ResNet_AE_CELEBA as Encoder_AE
from pythae.models.nn.benchmarks.celeba import Encoder_ResNet_SVAE_CELEBA as Encoder_SVAE
from pythae.models.nn.benchmarks.celeba import Encoder_ResNet_VAE_CELEBA as Encoder_VAE
from pythae.models.nn.benchmarks.celeba import Encoder_ResNet_VQVAE_CELEBA as Encoder_VQVAE

import os
from glob import glob

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

from plotting_utils import PlottingUtils

from dataloaders import get_CelebA_loader, get_PetExpression_loader, get_AltPet_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

BATCH_SIZE = 32
BATCH_SIZE_TEST = 32
NC = 3 # 1 # 
IMG_H = IMG_W = 64
EPOCHS = 1
DATA = 'Cifar'

#set up working data location
working_root = './'

if DATA == 'CelebA':
    data_root = './data/celeba/50k/'
    train_loader = get_CelebA_loader(data_root + "train.txt", BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment = True, shuffle = True, num_workers = 1, pin_memory = True) # Shape (C, H, W)
    eval_loader = get_CelebA_loader(data_root + "eval.txt", BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment = False, shuffle = True, num_workers = 1, pin_memory = True)
    test_loader = get_CelebA_loader(data_root + "test.txt", BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment = False, shuffle = True, num_workers = 1, pin_memory = True)
elif DATA == 'PetExpression'
    data_root = "./data/PetExpression/Master Folder/"
    train_loader = get_PetExpression_loader(data_root+'train/', batch_size=BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment=True, shuffle=True, num_workers=1, pin_memory=True)
    eval_loader = get_PetExpression_loader(data_root+'test/', batch_size=BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment=False, shuffle=False, num_workers=1, pin_memory=True)
elif DATA = 'Cifar':
    data_root = "./data/cifar/cifar10-64/"
    train_loader = get_PetExpression_loader(data_root+'train/', batch_size=BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment=True, shuffle=True, num_workers=1, pin_memory=True)
    eval_loader = get_PetExpression_loader(data_root+'test/', batch_size=BATCH_SIZE, shape = (NC, IMG_H, IMG_W), augment=False, shuffle=False, num_workers=1, pin_memory=True)
elif DATA == 'AltPet':
    data_root = './data/Altpets/'
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
    scheduler_cls='CosineAnnealingWarmRestarts',
    scheduler_params={'T_0': 50, 'eta_min':1e-6},
    #scheduler_cls='CyclicLR',
    #scheduler_params={'base_lr':1e-5, 'max_lr':0.001, 'mode':'triangular2','cycle_momentum':False},
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE_TEST,
    num_epochs=EPOCHS,
    no_cuda=False
)

model_config = VAEConfig(
    input_dim=(NC, IMG_H, IMG_W),
    latent_dim=128,
    reconstruction_loss="mse"
)
# Build the model
model = VAE(
    model_config=model_config,
    encoder=Encoder_VAE(model_config),
    decoder=Decoder_AE(model_config),
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

pu = PlottingUtils(trained_model, viz_size=2, data = next(iter(eval_loader))['data'].to(device))
pu.generated_from_samples(gen_data)
pu.original("height")
pu.original('img')
pu.reconstructed("height")
pu.reconstructed('img')
pu.show()
