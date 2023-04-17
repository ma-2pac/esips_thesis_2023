from nilmtk.api import API
import warnings
warnings.filterwarnings("ignore")
from nilmtk.disaggregate import CO
from nilmtk.disaggregate import FHMMExact, Mean


experiment = {
  'power': {'mains': ['active'],'appliance': ['active']},
  'sample_rate': 300,
  'appliances': ['dish washer','fridge'],
  'methods': {"FHMM_EXACT":FHMMExact({'num_of_states':2})},
  'train': {    
    'datasets': {
        'UKDALE': {
            'path': 'datasets/ukdale/ukdale2.h5',
            'buildings': {
                1: {
                    'start_time': '2013-03-01',
                    'end_time': '2013-05-01'
                    }
                }                
            }
        }
    },
  'test': {
    'datasets': {
        'UKDALE': {
            'path': 'datasets/ukdale/ukdale2.h5',
            'buildings': {
                1: {
                    'start_time': '2013-05-02',
                    'end_time': '2013-05-03'
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    },
    'display_predictions': True
}


api_results_experiment_2 = API(experiment)


