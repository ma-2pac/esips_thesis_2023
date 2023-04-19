import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from supervised_class_model.data_generator import DataGenerator
from supervised_class_model.model import build_model
from shared_files.utils import *
import importlib
import openpyxl
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#custom mods
import shared_files.dataset_utils as utils

if __name__=='__main__':
    path=''
    model_name='k_10e_1mon'
    test_sites="house1"
    appliance='kettle'

    # Choose the appliance-specific window size
    window_size = 100

    # Threshold of 15 Watt for detecting the ON/OFF states
    THRESHOLD = 15

    appliance_df, house_1, house_2, house_3, house_4, house_5 = utils.load_ukdale(path='datasets/', appliance=appliance)

    #merge different site data
    train_data=pd.DataFrame()
    test_data=pd.DataFrame()
    train_data=pd.concat([house_1,house_2,house_3,house_4,house_5])
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

    #slice between two dates
    train_start='2014-01-01'
    train_end='2014-02-01'
    test_start='2014-02-01'
    test_end='2014-02-03'

    train_data=train_data.loc[(train_data['time']>=train_start) & (train_data['time']<train_end)]
    test_data=test_data.loc[(test_data['time']>=test_start) & (test_data['time']<test_end)]

    #set validation size
    val_pert=0.1
    val_size=int(train_data.shape[0]*val_pert)



    main_train, appliance_train = np.array(train_data['agg_pow'][:-val_size]), np.array(train_data['app_pow'][:-val_size])
    main_val, appliance_val = np.array(train_data['agg_pow'][-val_size:]), np.array(train_data['app_pow'][-val_size:])
    main_test, appliance_test = np.array(test_data['agg_pow']), np.array(test_data['app_pow'])



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
    appliance_train_regression = normalize_data(appliance_train_regression, appliance_min_power, appliance_max_power)

    appliance_val_regression = np.copy(appliance_val)
    appliance_val_regression = normalize_data(appliance_val_regression, appliance_min_power, appliance_max_power)

    # Dataset generator
    batch_size = 8
    train_generator = DataGenerator(main_train, appliance_train_regression,
                                    appliance_train_classification, window_size, batch_size)
    val_generator = DataGenerator(main_val, appliance_val_regression,
                                    appliance_val_classification, window_size, batch_size)

    train_steps = train_generator.__len__()
    validation_steps = val_generator.__len__()

    # Tune the appliance-dependent parameters
    filters = 32
    kernel_size = 4
    units = 128

    model, att_model = build_model(window_size, filters, kernel_size, units)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(x=train_generator, epochs=10, steps_per_epoch=train_steps,
                        validation_data=val_generator, validation_steps=validation_steps,
                        callbacks=[early_stop], verbose=1,use_multiprocessing=True)

    # # Plotting the results of training
    # history_dict = history.history
    # plt.title('Loss during training')
    # plt.plot(np.arange(len(history.epoch)), history_dict['loss'])
    # plt.plot(np.arange(len(history.epoch)), history_dict['val_loss'])
    # plt.legend(['train', 'val'])
    # plt.show()

    #model = keras.models.load_model(f'saved_models/{model_name}')

    # Test
    appliance_test_classification = np.copy(appliance_test)
    appliance_test_classification[appliance_test_classification <= THRESHOLD] = 0
    appliance_test_classification[appliance_test_classification > THRESHOLD] = 1

    appliance_min_power = np.min(appliance_train)
    appliance_max_power = np.max(appliance_train)

    main_test_std = standardize_data(main_test, np.mean(main_test), np.std(main_test))

    appliance_test_regression = np.copy(appliance_test)
    appliance_test_regression = normalize_data(appliance_test_regression, appliance_min_power, appliance_max_power)

    batch_size = 32

    test_generator = DataGenerator(main_test_std, appliance_test_regression,
                                    appliance_test_classification, window_size, batch_size)

    test_steps = test_generator.__len__()

    results = model.evaluate(x=test_generator, steps=test_steps)
    predicted_output, predicted_on_off = model.predict(x=test_generator, steps=test_steps)

    predicted_output *= (appliance_max_power - appliance_min_power)
    predicted_output += appliance_min_power
    # Clip negative values to zero
    predicted_output[predicted_output < 0] = 0.0

    prediction = build_overall_sequence(predicted_output)
    prediction_on_off = build_overall_sequence(predicted_on_off)

    # Compute metrics
    N = 1200
    MAE = mae(prediction, appliance_test)
    SAE = sae(prediction, appliance_test, N=N)
    F1 = f1(prediction_on_off, appliance_test_classification)

    print("MAE = {}".format(MAE))
    print("SAE = {}".format(SAE))
    print("F1 = {}".format(F1))

    model.save(f'saved_models/{model_name}')
    '''
    Save Results
    '''
    #write sae, mae and f1 to text file
    with open(f"saved_results/{model_name}_test_{test_sites}.txt","w") as f:
        f.write(f"MAE: {MAE}\n")
        f.write(f"SAE: {SAE}\n")
        f.write(f"F1: {F1}\n")
        f.write(f"Start Date: {test_start}")
        f.write(f"End Date: {test_end}")

    '''
    Save CSV
    '''
    # #combine regression results
    # reg_df = pd.DataFrame()
    # reg_df = pd.concat([test_data['time'].reset_index(drop=True),pd.Series(appliance_test),pd.Series(prediction)],axis=1)
    # reg_df.columns=['time','actual','predicted']

    # #combine classification results
    # class_df = pd.DataFrame()
    # class_df = pd.concat([test_data['time'].reset_index(drop=True),pd.Series(appliance_test_classification),pd.Series(prediction_on_off)],axis=1)
    # class_df.columns=['time','actual','predicted']

    # #save csvs
    # reg_df.to_excel(f'saved_xlsx/{model_name}_reg.xlsx')
    # class_df.to_excel(f'saved_xlsx/{model_name}_class.xlsx')


    # # Plot the result of the prediction
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(50, 40))
    axes[0].set_title("Real Appliance")
    axes[0].plot(test_data['time'], appliance_test, color='blue')
    axes[1].set_title("Real Main")
    axes[1].plot(test_data['time'], main_test, color='orange')
    axes[2].set_title("Real vs prediction")
    axes[2].plot(test_data['time'], appliance_test, color='blue')
    axes[2].plot(test_data['time'], prediction, color='orange')
    axes[3].set_title("Real vs Prediction on off")
    axes[3].plot(test_data['time'], appliance_test_classification, color='blue')
    axes[3].plot(test_data['time'], prediction_on_off, color='orange')
    fig.tight_layout()
    plt.savefig(f"saved_results/{model_name}_test_{test_sites}.png")




