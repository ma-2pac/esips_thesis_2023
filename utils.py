'''
Utility functions to train different NILM models
'''

#std libs
import pandas as pd
import numpy as np

def load_train(THRESHOLD:int = 5, path: str = "dataset/", dataset="ukdale"):

    if dataset=="ukdale":
        #appliance data
        house_1_dw=pd.read_table(f'{path}ukdale/house_1/channel_6.dat',delimiter='\s+')
        house_1_fr=pd.read_table(f'{path}ukdale/house_1/channel_12.dat',delimiter='\s+')
        house_1_ke=pd.read_table(f'{path}ukdale/house_1/channel_10.dat',delimiter='\s+')
        house_1_mw=pd.read_table(f'{path}ukdale/house_1/channel_13.dat',delimiter='\s+')
        house_1_wm=pd.read_table(f'{path}ukdale/house_1/channel_5.dat',delimiter='\s+')

        house_3_ke = pd.read_table(f'{path}ukdale/house_3/channel_2.dat',delimiter='\s+')

        house_4_ke = pd.read_table(f'{path}ukdale/house_4/channel_3.dat',delimiter='\s+')


        house_5_ke = pd.read_table(f'{path}ukdale/house_5/channel_18.dat',delimiter='\s+')
        house_5_dw = pd.read_table(f'{path}ukdale/house_5/channel_22.dat',delimiter='\s+')
        house_5_mw = pd.read_table(f'{path}ukdale/house_5/channel_23.dat',delimiter='\s+')


        # get aggregate power signals for homes
        house_1=pd.read_table(f'{path}ukdale/house_1/channel_1.dat',delimiter='\s+')
        house_3=pd.read_table(f'{path}ukdale/house_3/channel_1.dat',delimiter='\s+')
        house_4=pd.read_table(f'{path}ukdale/house_4/channel_1.dat',delimiter='\s+')
        house_5=pd.read_table(f'{path}ukdale/house_5/channel_1.dat',delimiter='\s+')


        # rename columns
        house_1_ke.columns=['time','app_pow']
        house_3_ke.columns=['time','app_pow']
        house_4_ke.columns=['time','app_pow']
        house_5_ke.columns=['time','app_pow']

        house_1.columns=['time','agg_pow']
        house_3.columns=['time','agg_pow']
        house_4.columns=['time','agg_pow']
        house_5.columns=['time','agg_pow']

        # classify on state
        house_1_ke['on']=house_1_ke['app_pow']>THRESHOLD
        house_3_ke['on']=house_3_ke['app_pow']>THRESHOLD
        house_4_ke['on']=house_4_ke['app_pow']>THRESHOLD
        house_5_ke['on']=house_5_ke['app_pow']>THRESHOLD

        #merge aggregate and appliance data
        house_1_merge = pd.merge(house_1,house_1_ke,on='time',how='inner')
        house_3_merge = pd.merge(house_3,house_3_ke,on='time',how='inner')
        house_4_merge = pd.merge(house_4,house_4_ke,on='time',how='inner')
        house_5_merge = pd.merge(house_5,house_5_ke,on='time',how='inner')

        #merge different site data
        train_data=pd.concat([house_1_merge,house_3_merge,house_4_merge,house_5_merge])
        train_data.reset_index(inplace=True,drop=True)

        #get data at 5 min increment as time is in seconds
        train_data=train_data.iloc[::300]
        train_data.reset_index(inplace=True,drop=True)

        #standardise columns
        data_min= train_data['agg_pow'].min()
        data_max= train_data['agg_pow'].max()
        train_data['agg_pow']=standardize_data(train_data['agg_pow'],data_min=data_min,data_max=data_max)
        train_data['app_pow']=standardize_data(train_data['app_pow'],data_min=data_min,data_max=data_max)

        return train_data

#rename columns from dataset
def _rename_cols():
    pass

# standardise all data to be between scale 0-1
def standardize_data(data,data_min,data_max):

    data_scaled = (data - data_min) / (data_max - data_min)

    return data_scaled

def reshape_data(data):
    sequence_length = 10
    input_dim = 2
    num_samples = len(data) - sequence_length + 1

    x_train_reshaped = np.zeros((num_samples, sequence_length, input_dim))
    y_classification_train = np.zeros((num_samples, 1))
    y_reg_train = np.zeros((num_samples, 1))

    for i in range(num_samples):
        x_train_reshaped[i] = data.iloc[i:i+sequence_length, :2].values
        y_classification_train[i] = data.iloc[i+sequence_length-1, 2]
        y_reg_train[i] = data.iloc[i+sequence_length-1, 3]


    return x_train_reshaped, y_classification_train, y_reg_train


def preprocess_train():
    train_data = load_train()
    x_train, y_class_train, y_reg_train = reshape_data(train_data=train_data)

    return x_train, y_class_train, y_reg_train

def preprocess_test(THRESHOLD:int = 5, path: str = "dataset/", dataset="ukdale"):

    if dataset=="ukdale":
        house_2_ke = pd.read_table(f'{path}ukdale/house_2/channel_8.dat',delimiter='\s+')
        house_2=pd.read_table(f'{path}ukdale/house_2/channel_1.dat',delimiter='\s+')

        #rename columns
        house_2.columns=['time','agg_pow']
        house_2_ke.columns=['time','app_pow']

        #apply classification threshold
        house_2_ke['on']=house_2_ke['app_pow']>THRESHOLD

        #merge data and get 5 min increments
        test_data = pd.merge(house_2,house_2_ke,on='time',how='inner')
        test_data=test_data.iloc[::300]
        test_data.reset_index(inplace=True,drop=True)

        #standardise test data
        data_min= test_data['agg_pow'].min()
        data_max= test_data['agg_pow'].max()

        test_data['agg_pow']=standardize_data(test_data['agg_pow'],data_min=data_min,data_max=data_max)
        test_data['app_pow']=standardize_data(test_data['app_pow'],data_min=data_min,data_max=data_max)

        x_test, y_class_test, y_reg_test = reshape_data(test_data)

    return x_test, y_class_test, y_reg_test


