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


def load_refit(path:str,house: str, appliance: str):
    """_summary_

    Args:
        path (str): path to directory containing house csvs
        house (str): _description_
        appliance (str): _description_

    Raises:
        ValueError: _description_
    """

    house_df = pd.DataFrame()
    appliance_df=pd.DataFrame()

    if house == 'house_2':
        #get overall house df
        house_df = pd.read_csv(f'{path}/House_2.csv')
        if appliance=='fridge':
            appliance_df = house_df[['Unix','Aggregate','Appliance1']]
        elif appliance =='washing machine':
            appliance_df = house_df[['Unix','Aggregate','Appliance2']]
        elif appliance =='dishwasher':
            appliance_df = house_df[['Unix','Aggregate','Appliance3']]
        elif appliance =='microwave':
            appliance_df = house_df[['Unix','Aggregate','Appliance5']]
        elif appliance =='kettle':
            appliance_df = house_df[['Unix','Aggregate','Appliance8']]
        else:
            raise ValueError('Invalid appliance used')
        
            
    else:
        raise ValueError("Invalid house inputted. Format house name as house_{number}")
    
    #resample time column to 5 min
    #appliance_df['Time'] = pd.to_datetime(appliance_df['Time'])
    appliance_df=appliance_df.rename(columns={'Unix':'time'})
    appliance_df=_resample_time(appliance_df)
    
    return appliance_df


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

def train_test_split_ukdale(appliance,train_start,train_end,test_start,test_end):
    appliance_df, house_1, house_2, house_3, house_4, house_5 = load_ukdale(path='datasets/', appliance=appliance)

    #merge different site data
    train_data=pd.DataFrame()
    test_data=pd.DataFrame()
    train_data=pd.concat([house_1])
    train_data.reset_index(inplace=True,drop=True)

    #prep test data
    test_data = pd.concat([house_1])
    test_data.reset_index(inplace=True,drop=True)

    #reset headings
    train_data =train_data.rename(columns={'pow_x':'agg_pow', 'pow_y':'app_pow'})
    test_data =test_data.rename(columns={'pow_x':'agg_pow', 'pow_y':'app_pow'})


    train_data['agg_pow']=train_data['agg_pow'].astype(float)
    train_data['app_pow']=train_data['app_pow'].astype(float)
    test_data['agg_pow']=test_data['agg_pow'].astype(float)
    test_data['app_pow']=test_data['app_pow'].astype(float)

    train_data=train_data.loc[(train_data['time']>=train_start) & (train_data['time']<train_end)]
    test_data=test_data.loc[(test_data['time']>=test_start) & (test_data['time']<test_end)]

    #set validation size
    val_pert=0.1
    val_size=int(train_data.shape[0]*val_pert)


    main_train, appliance_train = np.array(train_data['agg_pow'][:-val_size]), np.array(train_data['app_pow'][:-val_size])
    main_val, appliance_val = np.array(train_data['agg_pow'][-val_size:]), np.array(train_data['app_pow'][-val_size:])
    main_test, appliance_test = np.array(test_data['agg_pow']), np.array(test_data['app_pow'])

    return main_train, appliance_train, main_val, appliance_val, main_test, appliance_test


def train_test_split_refit(appliance_df,train_start,train_end,test_start,test_end):

    # assign columns to be floats
    appliance_df['Aggregate']=appliance_df['Aggregate'].astype(float)
    appliance_df.iloc[:, -1]=appliance_df.iloc[:, -1].astype(float)



    #merge different site data
    train_data=pd.DataFrame()
    test_data=pd.DataFrame()
    
    train_data=pd.concat([appliance_df])
    train_data.reset_index(inplace=True,drop=True)

    #prep test data
    test_data = pd.concat([appliance_df])
    test_data.reset_index(inplace=True,drop=True)

    #set time ranges
    train_data=train_data.loc[(train_data['time']>=train_start) & (train_data['time']<train_end)]
    test_data=test_data.loc[(test_data['time']>=test_start) & (test_data['time']<test_end)]

    #set validation size
    val_pert=0.1
    val_size=int(train_data.shape[0]*val_pert)

    main_train, appliance_train = np.array(train_data['Aggregate'][:-val_size]), np.array(train_data.iloc[:, -1][:-val_size])
    main_val, appliance_val = np.array(train_data['Aggregate'][-val_size:]), np.array(train_data.iloc[:, -1][-val_size:])
    main_test, appliance_test = np.array(test_data['Aggregate']), np.array(test_data.iloc[:, -1])


    return main_train, appliance_train, main_val, appliance_val, main_test, appliance_test