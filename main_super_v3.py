import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from supervised_class_model.data_generator import DataGenerator
import supervised_class_model.model as nilm_models
from shared_files.utils import *
import importlib
import os
import random
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#custom mods
import shared_files.dataset_utils as utils


# Check if GPU is available
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


# set the random seeds of experiments to be exactly the same
seed_value = 12 #choose any value you want
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# ['dishwasher','kettle','microwave','washing machine','fridge']


#Setup initial parameters for testing
models_to_be_test = ['sgn','att']
appliance_list=['dishwasher','kettle','microwave','washing machine','fridge']
train_with_gan=False #set to be true if GAN synthetic data is to be used for training
train_model = True
train_with_refit = False 
test_sites="ukdale_house1"
window_size = 144 #enforce 144 for 12 hour window
epochs=5
THRESHOLD = 15 # Threshold of 15 Watt for detecting the ON/OFF states
ukdale_path = 'datasets/'
refit_path = 'datasets/refit'


# Tune the appliance-dependent parameters
filters = 64
kernel_size = 8
units = 1024
batch_size = 32 

for model_type in models_to_be_test:

    for app in appliance_list:

        
        model_name=f'{model_type}_{app}_{epochs}e_1year_{window_size}win'
        print(model_name)
        

        appliance_df, house_1, house_2, house_3, house_4, house_5 = utils.load_ukdale(path=ukdale_path, appliance=app)

        #merge different site data
        train_data=pd.concat([house_1])
        train_data.reset_index(inplace=True,drop=True)

        #prepare test data
        test_data = pd.concat([house_1])
        test_data.reset_index(inplace=True,drop=True)

        #reset columns
        train_data =train_data.rename(columns={'pow_x':'agg_pow', 'pow_y':'app_pow'})
        test_data =test_data.rename(columns={'pow_x':'agg_pow', 'pow_y':'app_pow'})

        #convert columns to float
        train_data['agg_pow']=train_data['agg_pow'].astype(float)
        train_data['app_pow']=train_data['app_pow'].astype(float)
        test_data['agg_pow']=test_data['agg_pow'].astype(float)
        test_data['app_pow']=test_data['app_pow'].astype(float)

        #slice between two dates
        train_start='2014-01-01'
        train_end='2015-01-01'
        test_start='2013-06-01'
        test_end='2013-06-08'

        train_data=train_data.loc[(train_data['time']>=train_start) & (train_data['time']<train_end)]
        test_data=test_data.loc[(test_data['time']>=test_start) & (test_data['time']<test_end)]

        #set validation size
        val_pert=0.1
        val_size=int(train_data.shape[0]*val_pert)

        #setup numpy arrays from dataframes
        main_train, appliance_train = np.array(train_data['agg_pow'][:-val_size]), np.array(train_data['app_pow'][:-val_size])
        main_val, appliance_val = np.array(train_data['agg_pow'][-val_size:]), np.array(train_data['app_pow'][-val_size:])


        """REFIT"""
        if train_with_refit:
            appliance_df = utils.load_refit(refit_path,'house_2', app) #use data from house 2 of refit
            appliance_df.reset_index(inplace=True,drop=True)

            #slice between two dates
            test_data=appliance_df.loc[(appliance_df['time']>=test_start) & (appliance_df['time']<test_end)]
            _, _, _, _, main_test, appliance_test = utils.train_test_split_refit(appliance_df,train_start,train_end,test_start,test_end)

        else:
            main_test, appliance_test = np.array(test_data['agg_pow']), np.array(test_data['app_pow'])


    
        #add padding to training numpy arrays
        main_train = np.pad(main_train, pad_width=int(window_size/2), mode='constant', constant_values=0)
        main_val = np.pad(main_val, pad_width=int(window_size/2), mode='constant', constant_values=0)
        appliance_train = np.pad(appliance_train, pad_width=int(window_size/2), mode='constant', constant_values=0)
        appliance_val = np.pad(appliance_val, pad_width=int(window_size/2), mode='constant', constant_values=0)


        #add additional training data if gan flag is used
        if train_with_gan:
            e=6000
            gan_data = pd.read_csv(f'datasets/refit/house_2/gan_{app}_{e}e_data.csv')

            #only keep time agg and app
            gan_data=gan_data.iloc[: , 1:]

            #convert to np
            main_train_gan, appliance_train_gan = np.array(gan_data['agg_pow']), np.array(gan_data['app_pow'])

            # add padding
            main_train_gan = np.pad(main_train_gan, pad_width=int(window_size/2), mode='constant', constant_values=0)
            appliance_train_gan = np.pad(appliance_train_gan, pad_width=int(window_size/2), mode='constant', constant_values=0)

            # add to existing train numpys
            main_train = np.append(main_train,main_train_gan)
            appliance_train = np.append(appliance_train,appliance_train_gan)

            #update model name
            model_name=f'gan_{model_type}_{app}_{epochs}e_1year_{window_size}win'
            print(model_name)


        if train_model:
            # Build ON/OFF appliance vector for the classification subtask
            appliance_train_classification = np.copy(appliance_train)
            appliance_train_classification[appliance_train_classification <= THRESHOLD] = 0
            appliance_train_classification[appliance_train_classification > THRESHOLD] = 1

            appliance_val_classification = np.copy(appliance_val)
            appliance_val_classification[appliance_val_classification <= THRESHOLD] = 0
            appliance_val_classification[appliance_val_classification > THRESHOLD] = 1

            # Standardization of the main power and normalization of appliance power
            appliance_min_power = np.min(appliance_train)
            appliance_max_power = np.max(appliance_train)
            main_std = np.std(main_train)
            main_mean = np.mean(main_train)

            main_train = standardize_data(main_train, np.mean(main_train), np.std(main_train))
            main_val = standardize_data(main_val, np.mean(main_val), np.std(main_val))

            appliance_train_regression = np.copy(appliance_train)
            #appliance_train_regression = normalize_data(appliance_train_regression, appliance_min_power, appliance_max_power)
            appliance_train_regression = standardize_data(appliance_train_regression, np.mean(appliance_train_regression), np.std(appliance_train_regression))

            appliance_val_regression = np.copy(appliance_val)
            #appliance_val_regression = normalize_data(appliance_val_regression, appliance_min_power, appliance_max_power)
            appliance_val_regression = standardize_data(appliance_val_regression, np.mean(appliance_val_regression), np.std(appliance_val_regression))

            # Dataset generator

            train_generator = DataGenerator(main_train, appliance_train_regression,
                                            appliance_train_classification, window_size, batch_size)
            val_generator = DataGenerator(main_val, appliance_val_regression,
                                            appliance_val_classification, window_size, batch_size)

            train_steps = train_generator.__len__()
            validation_steps = val_generator.__len__()

            #build the model based on the chosen model type
            if model_type =='att':
                model, att_model = nilm_models.build_att(window_size, filters, kernel_size, units)
            else:
                model = nilm_models.build_sgn(window_size, filters, kernel_size, units)

            #display summary
            model.summary()

            #setup early stopping machanism
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # fit the model
            history = model.fit(x=train_generator, epochs=epochs, steps_per_epoch=train_steps,
                                validation_data=val_generator, validation_steps=validation_steps,
                                callbacks=[early_stop], verbose=1,use_multiprocessing=False)
            # save model
            model.save(f'saved_models/{model_name}')
            if model_type =='att':
                att_model.save(f'saved_models/{model_name}_att_model')

            #Plotting the results of training
            history_dict = history.history
            plt.title('Loss during training')
            plt.plot(np.arange(len(history.epoch)), history_dict['loss'])
            plt.plot(np.arange(len(history.epoch)), history_dict['val_loss'])
            plt.legend(['train', 'val'])
            plt.show()


        # load saved model
        model = keras.models.load_model(f'saved_models/{model_name}')
        if model_type=='att':
            att_model = keras.models.load_model(f'saved_models/{model_name}_att_model')

        # Test setup
        appliance_test_classification = np.copy(appliance_test)
        appliance_test_classification[appliance_test_classification <= THRESHOLD] = 0
        appliance_test_classification[appliance_test_classification > THRESHOLD] = 1

        appliance_min_power = np.min(appliance_test)
        appliance_max_power = np.max(appliance_test)

        main_test_std = standardize_data(np.copy(main_test), np.mean(main_test), np.std(main_test))

        appliance_test_regression = np.copy(appliance_test)
        #appliance_test_regression = normalize_data(appliance_test_regression, appliance_min_power, appliance_max_power)
        appliance_test_regression = standardize_data(np.copy(appliance_test), np.mean(appliance_test), np.std(appliance_test))

        test_generator = DataGenerator(main_test_std, appliance_test_regression,
                                        appliance_test_classification, window_size, batch_size)

        test_steps = test_generator.__len__()

        results = model.evaluate(x=test_generator, steps=test_steps)
        predicted_output, predicted_on_off = model.predict(x=test_generator, steps=test_steps)

        #predicted_output *= (appliance_max_power - appliance_min_power)
        #predicted_output += appliance_min_power

        predicted_output = predicted_output*np.std(appliance_test)+np.mean(appliance_test)

        # Clip negative values to zero
        predicted_output[predicted_output < 10] = 0.0

        #scale predicted on off
        predicted_on_off[predicted_on_off > 0.05] = 1

    
        prediction = build_overall_sequence(predicted_output)
        prediction_on_off = build_overall_sequence(predicted_on_off)

        #attention test
        if model_type=='att':
            att_output = att_model.predict(x=test_generator, steps=test_steps)
            att_seq = build_overall_sequence(att_output)
            #att_seq = (att_seq - np.min(att_seq)) / (np.max(att_seq) - np.min(att_seq)) #normalise

        # Compute metrics
        N = 288
        MAE = mae(prediction, appliance_test)
        SAE = sae(prediction, appliance_test, N=N)
        F1 = f1(prediction_on_off, appliance_test_classification)

        print("MAE = {}".format(MAE))
        print("SAE = {}".format(SAE))
        print("F1 = {}".format(F1))


        '''
        Save Results
        '''
        #write sae, mae and f1 to text file
        with open(f"saved_results/{model_name}_test_{test_sites}.txt","w") as f:
            f.write(f"MAE: {MAE}\n")
            f.write(f"SAE: {SAE}\n")
            f.write(f"F1: {F1}\n")
            f.write(f"Start Date: {test_start}\n")
            f.write(f"End Date: {test_end}")

        #display aggregate consumption plot
        fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(10,5))
        axes.set_title("Real Main")
        plt.plot(test_data['time'], main_test, color='orange')
        plt.ylabel("Power (W)")
        plt.savefig(f"saved_results/{test_sites}_main.png",bbox_inches='tight')


        #Plot results 
        if model_type=='att':
        # Plot the result of the prediction
            fig, axes = plt.subplots(nrows=3, ncols=1,figsize=(13,10),gridspec_kw={'hspace': 0.35})
            axes[0].set_title("Real Appliance")
            axes[0].plot(test_data['time'], appliance_test, color='blue')
            axes[0].margins(x=0)
            axes[0].set_ylabel("Power (W)")
            # axes[1].set_title("Real vs Prediction on off")
            # axes[1].plot(test_data['time'], appliance_test_classification, color='blue')
            # axes[1].plot(test_data['time'], prediction_on_off, color='orange')
            #axes[1].margins(x=0)
            axes[1].set_title("Real vs prediction")
            axes[1].plot(test_data['time'], appliance_test, color='blue')
            axes[1].plot(test_data['time'], prediction, color='orange')
            axes[1].margins(x=0)
            axes[1].set_ylabel("Power (W)")
            axes[2].set_title("Attention Weights")
            #axes[3].plot(test_data['time'].iloc[:att_seq.shape[0]], att_seq, color='blue')
            #axes[3].margins(x=0)
            cax = axes[2].pcolormesh([att_seq], cmap='plasma', shading='auto')
            fig.colorbar(cax,ax=axes[2],location='bottom')
        else:
            fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(13,10),gridspec_kw={'hspace': 0.35})
            axes[0].set_ylabel("Power (W)")
            axes[0].set_title("Real Appliance")
            axes[0].plot(test_data['time'], appliance_test, color='blue')
            # axes[1].set_title("Real vs Prediction on off")
            # axes[1].plot(test_data['time'], appliance_test_classification, color='blue')
            # axes[1].plot(test_data['time'], prediction_on_off, color='orange')
            axes[1].set_title("Real vs prediction")
            axes[1].plot(test_data['time'], appliance_test, color='blue')
            axes[1].plot(test_data['time'], prediction, color='orange')
            axes[1].set_ylabel("Power (W)")

        plt.savefig(f"saved_results/{model_name}_test_{test_sites}_full.png",bbox_inches='tight')





