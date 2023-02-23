#std libs
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import importlib

#custom mods
import utils
import models

#reload mods
importlib.reload(utils)
importlib.reload(models)

#initial params
# Choose the appliance-specific window size
window_size = 512

# Threshold of 15 Watt for detecting the ON/OFF states
THRESHOLD = 15

x_train, y_class_train, y_reg_train = utils.preprocess_train(THRESHOLD=THRESHOLD)
x_test, y_class_test, y_reg_test = utils.preprocess_test(THRESHOLD=THRESHOLD)

# Tune the appliance-dependent parameters
filters = 32
kernel_size = 4
units = 128

model, att_model = models.build_attention_model(window_size, filters, kernel_size, units)
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(x=x_train, epochs=100, steps_per_epoch=len(x_train),
                    validation_data=val_generator, validation_steps=validation_steps,
                    callbacks=[early_stop], verbose=1)
