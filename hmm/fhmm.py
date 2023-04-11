'''
Based of code from https://github.com/beckel/nilm-eval/blob/master/Python/fhmm.py
'''

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import hmm
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
import itertools
from copy import deepcopy



