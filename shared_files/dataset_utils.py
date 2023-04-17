'''
Functions associated with bringing in and preprocessing dataset data
'''

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
#from nilmtk.dataset_converters import convert_ukdale

def load_ukdale(path: str, appliance: str):

    column_names=['time', 'pow']
    
    #appliance data
    appliance_df=pd.DataFrame()
    house_1=pd.DataFrame()
    house_2=pd.DataFrame()
    house_3=pd.DataFrame()
    house_4=pd.DataFrame()
    house_5=pd.DataFrame()


    # get aggregate power signals for homes
    house_1_agg=_resample_time(pd.read_table(f'{path}ukdale/house_1/channel_1.dat',delimiter='\s+',header=None,names=column_names))
    house_2_agg=_resample_time(pd.read_table(f'{path}ukdale/house_2/channel_1.dat',delimiter='\s+',header=None,names=column_names))
    house_3_agg=_resample_time(pd.read_table(f'{path}ukdale/house_3/channel_1.dat',delimiter='\s+',header=None,names=column_names))
    house_4_agg=_resample_time(pd.read_table(f'{path}ukdale/house_4/channel_1.dat',delimiter='\s+',header=None,names=column_names))
    house_5_agg=_resample_time(pd.read_table(f'{path}ukdale/house_5/channel_1.dat',delimiter='\s+',header=None,names=column_names))

    if appliance=='dishwasher':

        #load house 1 data
        house_1_app = _resample_time(pd.read_table(f'{path}ukdale/house_1/channel_6.dat',delimiter='\s+',header=None,names=column_names))
        house_2_app = _resample_time(pd.read_table(f'{path}ukdale/house_2/channel_13.dat',delimiter='\s+',header=None,names=column_names))
        house_5_app = _resample_time(pd.read_table(f'{path}ukdale/house_5/channel_22.dat',delimiter='\s+',header=None,names=column_names))

        #merge with aggregates
        house_1 = pd.merge(house_1_agg,house_1_app,on='time',how='inner')
        house_2 = pd.merge(house_2_agg,house_2_app,on='time',how='inner')
        house_5 = pd.merge(house_5_agg,house_5_app,on='time',how='inner')

        appliance_df=pd.concat([house_1,house_2,house_5])

    elif appliance=='fridge':
        house_1_app=_resample_time(pd.read_table(f'{path}ukdale/house_1/channel_12.dat',delimiter='\s+',header=None,names=column_names))
        house_2_app=_resample_time(pd.read_table(f'{path}ukdale/house_2/channel_14.dat',delimiter='\s+',header=None,names=column_names))

        #merge with aggregates
        house_1 = pd.merge(house_1_agg,house_1_app,on='time',how='inner')
        house_2 = pd.merge(house_2_agg,house_2_app,on='time',how='inner')

        appliance_df=pd.concat([house_1,house_2])


    elif appliance=='kettle':
        house_1_app =_resample_time(pd.read_table(f'{path}ukdale/house_1/channel_10.dat',delimiter='\s+',header=None,names=column_names))
        house_2_app =_resample_time(pd.read_table(f'{path}ukdale/house_2/channel_8.dat',delimiter='\s+',header=None,names=column_names))
        house_3_app = _resample_time(pd.read_table(f'{path}ukdale/house_3/channel_2.dat',delimiter='\s+',header=None,names=column_names))
        house_4_app = _resample_time(pd.read_table(f'{path}ukdale/house_4/channel_3.dat',delimiter='\s+',header=None,names=column_names))
        house_5_app = _resample_time(pd.read_table(f'{path}ukdale/house_5/channel_18.dat',delimiter='\s+',header=None,names=column_names))

        #merge with aggregates
        house_1 = pd.merge(house_1_agg,house_1_app,on='time',how='inner')
        house_2 = pd.merge(house_2_agg,house_2_app,on='time',how='inner')
        house_3 = pd.merge(house_3_agg,house_3_app,on='time',how='inner')
        house_4 = pd.merge(house_4_agg,house_4_app,on='time',how='inner')
        house_5 = pd.merge(house_5_agg,house_5_app,on='time',how='inner')

        appliance_df=pd.concat([house_1,house_2,house_3, house_4, house_5])


    elif appliance=='microwave':
        house_1_app = _resample_time(pd.read_table(f'{path}ukdale/house_1/channel_13.dat',delimiter='\s+',header=None,names=column_names))
        house_2_app = _resample_time(pd.read_table(f'{path}ukdale/house_2/channel_15.dat',delimiter='\s+',header=None,names=column_names))
        house_5_app = _resample_time(pd.read_table(f'{path}ukdale/house_5/channel_23.dat',delimiter='\s+',header=None,names=column_names))

        #merge with aggregates
        house_1 = pd.merge(house_1_agg,house_1_app,on='time',how='inner')
        house_2 = pd.merge(house_2_agg,house_2_app,on='time',how='inner')
        house_5 = pd.merge(house_5_agg,house_5_app,on='time',how='inner')

        appliance_df=pd.concat([house_1,house_2, house_5])

    elif appliance=='washing machine':
        house_1_app = _resample_time(pd.read_table(f'{path}ukdale/house_1/channel_5.dat',delimiter='\s+',header=None,names=column_names))
        house_2_app = _resample_time(pd.read_table(f'{path}ukdale/house_2/channel_12.dat',delimiter='\s+',header=None,names=column_names))
        house_5_app = _resample_time(pd.read_table(f'{path}ukdale/house_5/channel_24.dat',delimiter='\s+',header=None,names=column_names))

        #merge with aggregates
        house_1 = pd.merge(house_1_agg,house_1_app,on='time',how='inner')
        house_2 = pd.merge(house_2_agg,house_2_app,on='time',how='inner')
        house_5 = pd.merge(house_5_agg,house_5_app,on='time',how='inner')

        appliance_df=pd.concat([house_1,house_2, house_5])

    else:
        raise ValueError("Invalid appliance chosen")


    print("loaded appliance data")

    return appliance_df, house_1, house_2, house_3, house_4, house_5


#resample dataset from given timestep
def _resample_time(df):
    #change from seconds to timestamp
    df['time'] = df['time'].apply(lambda x: dt.datetime.fromtimestamp(x))

    #remove seconds and round down to nearest minute
    df['time'] = df['time'].dt.floor('1min')

    #only take times at 5min increments
    df = df[df['time'].dt.minute % 5 == 0]

    #only take the first instance of capture in a 5 min window
    df = df.drop_duplicates(subset=['time'], keep='first')

    return df

#convert the ukdale dataset into h5 format
def convert_ukdale_h5(path):
    
    convert_ukdale(ukdale_path=path,output_filename='../datasets/ukdale/ukdale2.h5')
