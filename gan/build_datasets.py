'''
functions associated with building datasets from gan generators
'''

#libs
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler as scale_data
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False
 
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def preprocess_house_data(house_df: pd.DataFrame, start_time, end_time):

    new_house_df =house_df.copy()
    new_house_df = new_house_df.rename(columns={'pow_x':'agg_pow', 'pow_y':'app_pow'})

    #assign type
    new_house_df['agg_pow']=new_house_df['agg_pow'].astype(float)
    new_house_df['app_pow']=new_house_df['app_pow'].astype(float)

    #slice between time
    new_house_df=new_house_df.loc[(new_house_df['time']>=start_time) & (new_house_df['time']<end_time)]

    #return house
    return new_house_df

def build_gan_dataset(generator: nn.Module, house_df: pd.DataFrame, output_path,file_name, window_size):

    #take copy of  load and make numpy array
    agg_np = np.array(house_df['agg_pow'])
    app_np = np.array(house_df['app_pow'])
    
    #get dimensions for reshaping
    len = agg_np.shape[0]-(agg_np.shape[0] % window_size)
    seg_num=int(len/window_size)

    #reshape 1d vector into 2D vector 
    agg_np =agg_np[:len].reshape((seg_num,window_size))
    app_np =app_np[:len].reshape((seg_num,window_size))

    # Scaling data
    scale_agg=scale_data()
    scale_app=scale_data()

    scale_agg.fit(agg_np)
    scale_app.fit(app_np)

    # Transforming aggregate data to be column vector
    X_np=scale_agg.transform(agg_np)

    a_to_s=X_np 
    agg_to_sample=a_to_s.reshape(a_to_s.shape[0],1,a_to_s.shape[1])

    #get a  generated sample from the generator
    agg_to_sample = Variable(FloatTensor(agg_to_sample))
    gen_d = generator(agg_to_sample)

    # transform data back to original scale
    gen_ds_inv=scale_app.inverse_transform(gen_d.cpu().detach())

    #make data one long string
    gen_ds=gen_ds_inv.reshape(-1)

    #slice df so that it is same length as generated array
    house_df=house_df.head(gen_ds.shape[0])

    #replace appliance power in house df with generated data
    house_df['app_pow']=gen_ds

    #save df as csv
    house_df.to_csv(f'{output_path}/{file_name}.csv')

    return house_df
