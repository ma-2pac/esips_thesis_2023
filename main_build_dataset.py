import argparse
import os
import numpy as np
import math
import importlib

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler as scale_data

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt

#custom mods
import shared_files.dataset_utils as d_utils
import gan.build_datasets as bd

from gan.gan_model import Generator

importlib.reload(bd)

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Arguments:
    def __init__(self) -> None:
        self.n_epochs=6000
        self.batch_size=1024
        self.lr=0.00001
        self.b1=0.5
        self.b2=0.999
        self.n_cpu=8
        self.latent_dim=100
        self.img_size=28
        self.channels=1
        self.sample_interval=400


opt=Arguments()
"""
dataset flag
"""
is_ukdale=False



"""
Load UKDALE Data
"""
appliance_win_dict={
    'dishwasher':int(1536*6/300),
    'fridge':int(5126*6/300),
    'kettle':int(128*6/300),
    'microwave':int(288*6/300),
    'washing machine':int(1024*6/300)
}

appliance_list=['dishwasher','kettle','microwave','washing machine','fridge']

for app in appliance_list:

    #set start and end time to generate training data
    train_start='2014-01-01'
    train_end='2014-07-01'
    test_start='2014-01-01'
    test_end='2014-07-01'

    # Choose the appliance-specific window size
    window_size = appliance_win_dict[app]
    if window_size<50:
        window_size=50

    window_size=144


    model_name=f'gan_{app}_{opt.n_epochs}e_1year_{window_size}win'
    print(model_name)
    appliance=app

    if is_ukdale:
        pass
    else:
        appliance_df = d_utils.load_refit('datasets/refit','house_2', app)
        appliance_df.reset_index(inplace=True,drop=True)
        appliance_df = appliance_df.rename(columns={'Aggregate':'pow_x', appliance_df.columns[-1]:'pow_y'})

        house_df=bd.preprocess_house_data(appliance_df,train_start,train_end)

    '''
    Generation of synthetic appliance data after generator has been trained
    '''
    # Initialize generator and discriminator
    generator = Generator(window_size)

    if cuda:
        generator.cuda()

    generator.load_state_dict(torch.load(f"saved_models/{model_name}_generator.pth"))

    print("generator loaded")

    generated_df=bd.build_gan_dataset(generator=generator,house_df=house_df,output_path=f'datasets/refit/house_2/',file_name=f'gan_{app}_{opt.n_epochs}e_data',window_size=window_size)
