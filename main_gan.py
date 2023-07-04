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
import gan.gan_model as gan 

importlib.reload(gan)
importlib.reload(bd)
importlib.reload(d_utils)



os.makedirs("gan_loads", exist_ok=True)

# parser = argparse.ArgumentParser()
# parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
# opt = parser.parse_args()
# print(opt)

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

def smooth(y, box_pts=2):
#     print("smooth in",y.shape)
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
#     print("smooth out",y_smooth.shape)
    return y_smooth


opt=Arguments()

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

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
#appliance_list=['kettle']



#app=appliance_list[0]
for app in appliance_list:

    # Choose the appliance-specific window size
    window_size = appliance_win_dict[app]
    if window_size<50:
        window_size=50

    # enforce window size to be 12 hours from now on
    window_size = 144

    #window_size =50 

    model_name=f'gan_{app}_{opt.n_epochs}e_1year_{window_size}win'
    print(model_name)
    test_sites="house2"

    # Threshold of 15 Watt for detecting the ON/OFF states
    THRESHOLD = 15
    train_start='2014-01-01'
    train_end='2015-01-01'
    test_start='2013-06-01'
    test_end='2013-06-08'

    main_train, appliance_train, main_val, appliance_val, main_test, appliance_test=d_utils.train_test_split_ukdale(app,train_start,train_end,test_start,test_end)



    #pad train arrays with 0s on either side of array of  length window_size/2 each
    main_train = np.pad(main_train, (window_size//2, window_size//2), mode='constant', constant_values=0)
    appliance_train = np.pad(appliance_train, (window_size//2, window_size//2), mode='constant', constant_values=0)

    #get dimensions for reshaping
    # train_len = main_train.shape[0]-(main_train.shape[0] % window_size)
    # seg_num=int(train_len/window_size)

    #reshape 1d vector into 2D vector comprised of segments from a sliding window
    main_train_r =np.array([main_train[i-window_size//2:i+window_size//2] for i in range(window_size//2,len(main_train)-window_size//2)])
    appliance_train_r =np.array([appliance_train[i-window_size//2:i+window_size//2] for i in range(window_size//2,len(appliance_train)-window_size//2)])

    window_size=appliance_train_r.shape[1]

    # Scaling data
    scale_agg=scale_data()
    scale_app=scale_data()

    scale_agg.fit(main_train_r)
    scale_app.fit(appliance_train_r)

    # Transforming training data to be column vector
    X_train_n=scale_agg.transform(main_train_r)
    Y_train_n=scale_app.transform(appliance_train_r)


    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    feature_loss=torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = gan.Generator(window_size)
    discriminator = gan.Discriminator(window_size)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    class DataIterator(Dataset):
        def __init__(self,X_train_n):

            self.len=X_train_n.shape[0]
            self.xdata=torch.from_numpy(X_train_n)
            self.ydata=torch.from_numpy(Y_train_n)
            
        def __getitem__(self,index):
            return (self.xdata[index],self.ydata[index])
        
        def __len__(self):
            return(self.len)

    train_it=DataIterator(X_train_n)
    dataloader=DataLoader(dataset=train_it,batch_size=opt.batch_size,shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    # g_losses=[]
    # d_losses=[]

    # for epoch in range(opt.n_epochs):
    #     g_running_loss=0.0
    #     d_running_loss=0.0
    #     for i, (aggregate, appliance) in enumerate(dataloader):

    #         #reshaping to fit the convolution layer
    #         aggregate_reshaped=aggregate.reshape(aggregate.shape[0],1,aggregate.shape[1])

    #         batch_size = appliance.shape[0]
    #         app_np=appliance[:,:]

    #         # Adversarial ground truths
    #         valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
    #         fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

    #         # Configure input
    #         real_aggregate_reshaped = Variable(aggregate_reshaped.type(FloatTensor))
    #         real_appliance = Variable(app_np.type(FloatTensor))
    #         real_aggregate= Variable(aggregate.type(FloatTensor))

    #         real_app_real_agg=torch.cat((real_appliance,real_aggregate),1)

    #         # -----------------
    #         #  Train Generator
    #         # -----------------

    #         #set gradients of all parameters of generator to be 0
    #         optimizer_G.zero_grad()

    #         #Use aggregate load as sample noise, which in turn generates appliance data that is tested by the discriminator
    #         gen_appliance = generator(real_aggregate_reshaped)

    #         #concatenate real aggregate with generated appliance
    #         gen_app_real_agg=torch.cat((gen_appliance,real_aggregate),1)

    #         validity,feat = discriminator(gen_app_real_agg)

    #         # Loss measures generator's ability to fool the discriminator
    #         # g_loss = adversarial_loss(validity, valid)
    #         g_loss=feature_loss(feat, real_app_real_agg)

    #         #computers the gradients of the generator's parameters with respect to the loss function
    #         g_loss.backward()

    #         #updates the parameters of generator based on gradients computed from back propagation
    #         optimizer_G.step()

    #         g_losses.append(g_loss.item())


    #         # ---------------------
    #         #  Train Discriminator
    #         # ---------------------

    #         optimizer_D.zero_grad()

    #         #concatenate appliance and aggregate
    #         real_app_real_agg=torch.cat((appliance,aggregate),1)

    #         #create tensor object
    #         real_app_real_agg=Variable(real_app_real_agg.type(FloatTensor))

    #         #get validity from real load profiles
    #         validity_real,_ = discriminator(real_app_real_agg)
    #         d_real_loss = adversarial_loss(validity_real, valid)

    #         #get validity from fake load profiles
    #         validity_fake,_ = discriminator(gen_app_real_agg.detach()) #detach
    #         d_fake_loss = adversarial_loss(validity_fake, fake)

    #         #determine discriminator loss
    #         d_loss = (d_real_loss + d_fake_loss) / 2


    #         d_loss.backward()
    #         optimizer_D.step()

    #         d_losses.append(d_loss.item())




    #         batches_done = epoch * len(dataloader) + i
    #         if batches_done % opt.sample_interval == 0:
    #             print("Batch_no:",batches_done," Epoch:",epoch)

    #             # #take a snippet from the training data
    #             # a_to_s=X_train_n[0:10,:] # z
    #             # agg_to_sample=a_to_s.reshape(a_to_s.shape[0],1,a_to_s.shape[1])
                
    #             # #get the true appliance data
    #             # true_app=Y_train_n[0:10,:]
            
    #             # #get a test generated sample from the generator
    #             # agg_to_sample = Variable(FloatTensor(agg_to_sample))
    #             # gen_d = generator(agg_to_sample)
                

    #             # # transform data back to original scale
    #             # gen_ds_inv=scale_app.inverse_transform(gen_d.cpu().detach())
    #             # true_app_inv=scale_app.inverse_transform(true_app)
            
                
            
    #             # plt.plot(smooth(gen_ds_inv[0,:]),label="GAN")
    #             # plt.plot(true_app_inv[0,:],label="Real")
    #             # plt.legend()
    #             # plt.show()

    #             # plt.plot(smooth(gen_ds_inv[5,:]),label="GAN")
    #             # plt.plot(true_app_inv[5,:],label="real")
    #             # plt.legend()
    #             # plt.show()

    # print(
    #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
    #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
    # )
    
    # plt.plot(d_losses,label="discriminator")
    # plt.plot(g_losses,label="generator")
    # plt.legend()
    # plt.show()


    # Save the generator and discriminator to disk
    #torch.save(generator.state_dict(), f"saved_models/{model_name}_generator.pth")
    #torch.save(discriminator.state_dict(), f"saved_models/{model_name}_discriminator.pth")

    '''
    Generation of synthetic appliance data after generator has been trained
    '''
    # # Initialize generator and discriminator
    # generator = gan.Generator(window_size)
    # discriminator = gan.Discriminator(window_size)

    # if cuda:
    #     generator.cuda()
    #     discriminator.cuda()
    #     adversarial_loss.cuda()
    generator.load_state_dict(torch.load(f"saved_models/{model_name}_generator.pth"))
    discriminator.load_state_dict(torch.load(f"saved_models/{model_name}_discriminator.pth"))

    #get dimensions for reshaping
    test_len = main_test.shape[0]-(main_test.shape[0] % window_size)
    seg_num=int(test_len/window_size)

    #reshape 1d vector into 2D vector 
    main_test_r =main_test[:test_len].reshape((seg_num,window_size))
    appliance_test_r =appliance_test[:test_len].reshape((seg_num,window_size))

    window_size=appliance_test_r.shape[1] 

    # Scaling data
    scale_agg=scale_data()
    scale_app=scale_data()

    scale_agg.fit(main_test_r)
    scale_app.fit(appliance_test_r)

    # Transforming training data to be column vector
    X_test_n=scale_agg.transform(main_test_r)
    Y_test_n=scale_app.transform(appliance_test_r)


    a_to_s=X_test_n 
    agg_to_sample=a_to_s.reshape(a_to_s.shape[0],1,a_to_s.shape[1])
    true_app=Y_test_n


    #get a test generated sample from the generator
    agg_to_sample = Variable(FloatTensor(agg_to_sample))
    gen_d = generator(agg_to_sample)

    # transform data back to original scale
    gen_ds_inv=scale_app.inverse_transform(gen_d.cpu().detach())
    true_app_inv=scale_app.inverse_transform(true_app)
    true_agg_inv=scale_agg.inverse_transform(a_to_s)

    #make data one long string
    true_agg=true_agg_inv.reshape(-1)
    true_app=true_app_inv.reshape(-1)
    gen_ds=gen_ds_inv.reshape(-1)



    # plt.plot(true_agg_inv[0,:],label="Agg")
    # plt.legend()
    # plt.show()

    # plt.plot(smooth(gen_ds_inv[0,:]),label="GAN")
    # plt.plot(true_app_inv[0,:],label="Real")
    # plt.legend()
    # plt.show()

    # plt.plot(smooth(gen_ds_inv[5,:]),label="GAN")
    # plt.plot(true_app_inv[5,:],label="real")
    # plt.legend()
    # plt.show()

    # # Plot the result of the prediction
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(50, 40))
    axes[0].set_title("Real Appliance")
    axes[0].plot(true_app, color='blue')
    axes[1].set_title("Real Main")
    axes[1].plot(true_agg, color='orange')
    axes[2].set_title("GAN")
    axes[2].plot(smooth(gen_ds), color='blue')
    fig.tight_layout()
    #plt.savefig(f"saved_results/{model_name}_test_{test_sites}.png")

    '''
    Build dataset for appliance
    '''
    #set start and end time to generate training data
    train_start='2013-04-18'
    train_end='2013-10-09'

    #appliance_df, house_1, house_2, house_3, house_4, house_5 = d_utils.load_ukdale(path='datasets/', appliance=app)

    appliance_df = d_utils.load_refit('datasets/refit','house_2', app)
    appliance_df.reset_index(inplace=True,drop=True)
    #set time ranges
    train_data=appliance_df.loc[(appliance_df['time']>=train_start) & (appliance_df['time']<train_end)]
    test_data=appliance_df.loc[(appliance_df['time']>=test_start) & (appliance_df['time']<test_end)]

    main_train, appliance_train, main_val, appliance_val, main_test, appliance_test = d_utils.train_test_split_refit(appliance_df,train_start,train_end,test_start,test_end)


    house_df=bd.preprocess_house_data(house_2,train_start,train_end)

    generated_df=bd.build_gan_dataset(generator=generator,house_df=house_df,output_path=f'datasets/ukdale/house_2/',file_name=f'gan_{app}_{opt.n_epochs}e_data',window_size=window_size)


