#libraries
import importlib
from nilmtk import DataSet
from nilmtk.disaggregate import fhmm_exact
from nilmtk.utils import print_dict
import shared_files.utils as utils
import shared_files.dataset_utils as ds_utils

#reload
importlib.reload(utils)
importlib.reload(ds_utils)


if __name__ == '__main__':

    #load ukdale dataset in h5
    uk_dale =DataSet('datasets/ukdale/ukdale.h5')

    # Set the sample period and resample the data if needed
    sample_period = 300  # Set the sample period in seconds
    uk_dale.set_window(start='2013-05-01', end='2013-05-07')  # Set the time window for training

    # Load mains and appliance data
    mains = uk_dale.buildings[1].elec.mains()  # Replace with the appropriate building number
    appliances = uk_dale.buildings[1].elec.submeters()  # Replace with the appropriate building number

    params={'save-model-path':'hmm',
            'pretrained-model-path': None,
            'chunk_wise_training': False,
            'num_of_states':2}

    # Train the FHMM model
    fhmm = fhmm_exact.FHMMExact(params=params)
    fhmm.partial_fit(mains, appliances, sample_period=sample_period)

    # Print learned model parameters
    print_dict(fhmm.model)

    # Save the trained model to a file
    fhmm.export_model('hmm/fhmm_model.pkl')  # Replace with the desired file path

