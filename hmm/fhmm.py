'''
Based of code from https://github.com/beckel/nilm-eval/blob/master/Python/fhmm.py
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
import itertools
from copy import deepcopy
from hmmlearn import hmm
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split



# function to build the factorial hidden markov model
def train_fhmm(data: pd.DataFrame, n_states: int, n_chains: int, n_features: int):
    # Initialize the FHMM model
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=100)

    # Train the FHMM model
    model.fit(data)

    # Generate samples from the trained FHMM model
    generated_samples, _ = model.sample(n_samples=100)  # Specify the number of samples to generate

    # Perform inference on a new sequence using the trained FHMM model
    new_sequence = np.array([0, 1, 2, 0, 1, 2])  # Example new sequence
    log_prob, state_sequence = model.decode(new_sequence.reshape(-1, 1))

    # Print the generated samples and the inferred state sequence
    print("Generated samples:", generated_samples)
    print("Inferred state sequence:", state_sequence)


def train_hmm(X_train,n_states: int):
    # Define the FHMM model
    model = GaussianHMM(n_components=n_states, covariance_type="full")

    model.fit(X_train)

