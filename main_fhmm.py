from nilmtk.api import API
import warnings
warnings.filterwarnings("ignore")
from nilmtk.disaggregate import CO
from nilmtk.disaggregate import FHMMExact, Mean
import pandas as pd
import numpy as np
from shared_files.utils import *
from nilmtk.dataset_converters import convert_refit
import os
import random

'''
convert datasets
'''

convert_refit(input_path=r'C:\Users\marco\OneDrive\Documents\GitHub\esips_thesis_2023\datasets\refit',output_filename='datasets/refit/refit.h5')

# Set seed value
seed_value = 12 #choose any value you want
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)


#setup experiment parameters for the NILMTK API
experiment = {
  'power': {'mains': ['active'],'appliance': ['active']},
  'sample_rate': 300, #5 minutes
  'appliances': ['dish washer','fridge','kettle', 'washing machine', 'microwave'],
  'methods': {"FHMM_EXACT":FHMMExact({'num_of_states':2})},
  'train': {    
    'datasets': {
        'UKDALE': {
            'path': 'datasets/ukdale/ukdale2.h5',
            'buildings': {
                1: {
                    'start_time': '2014-01-01',
                    'end_time': '2015-01-01'
                    }
                }                
            }
        }
    },
  'test': {
    'datasets': {
        'UKDALE': {
            'path': 'datasets/refit/refit.h5',
            'buildings': {
                2: {
                    'start_time': '2013-10-01',
                    'end_time': '2013-10-08'
                    }
                }
            }
        },
        'metrics':['mae', 'rmse','sae']
    },
    'display_predictions': True
}

#run the experiment
api_results_experiment = API(experiment)

#calculate F1

gt_df=api_results_experiment.gt_overall #ground truth
pred_df=api_results_experiment.pred_overall['FHMM_EXACT'] #prediction

#set threshold
THRESHOLD = 15

#loop through appliances
for app in pred_df.columns:


    #set classification
    appliance_test_classification = np.array(pred_df[app])
    appliance_test_classification[appliance_test_classification <= THRESHOLD] = 0
    appliance_test_classification[appliance_test_classification > THRESHOLD] = 1

    appliance_gt_classification = np.array(gt_df[app])
    appliance_gt_classification[appliance_gt_classification <= THRESHOLD] = 0
    appliance_gt_classification[appliance_gt_classification > THRESHOLD] = 1

    #calculate f1 score
    F1 = f1(appliance_gt_classification, appliance_test_classification)
    print(f"F1 {app}: {F1}")

    #calculte SAE
    N = 288
    SAE = sae(pred_df[app], gt_df[app], N=N)
    print(f"SAE {app}: {SAE}\n")





