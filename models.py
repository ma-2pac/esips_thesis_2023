'''
Functions to create different models for NILM
'''

#std libs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

from attention_layer import AttentionLayer

def build_LSTM(window_size,features,units):
    model = Sequential()
    model.add(InputLayer(input_shape=(window_size, features)))
    model.add(LSTM(units=units,return_sequences=True))
    model.add(LSTM(units=units))
    model.add(Dense(1, activation='sigmoid'))

    cp1 = ModelCheckpoint('model/', save_best_only=True)

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    return model

def build_attention_model(window_size, filters, kernel_size, units):
    input_data = tf.keras.Input(shape=(window_size, 1))

    # CLASSIFICATION SUBNETWORK
    x = tf.keras.layers.Conv1D(filters=30, kernel_size=10, activation='relu')(input_data)
    x = tf.keras.layers.Conv1D(filters=30, kernel_size=8, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=40, kernel_size=6, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer='he_normal')(x)
    classification_output = tf.keras.layers.Dense(units=window_size, activation='sigmoid', name="classification_output")(x)

    #REGRESSION SUBNETWORK
    y = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(input_data)
    y = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(y)
    y = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(y)
    y = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(y)
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, activation="tanh", return_sequences=True), merge_mode="concat")(y)
    y, weights = AttentionLayer(units=units)(y)
    y = tf.keras.layers.Dense(units, activation='relu')(y)
    regression_output = tf.keras.layers.Dense(window_size, activation='relu', name="regression_output")(y)

    output = tf.keras.layers.Multiply(name="output")([regression_output, classification_output])

    full_model = tf.keras.Model(inputs=input_data, outputs=[output, classification_output], name="LDwA")
    attention_model = tf.keras.Model(inputs=input_data, outputs=weights)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    full_model.compile(optimizer=optimizer, loss={
        "output": tf.keras.losses.MeanSquaredError(),
        "classification_output": tf.keras.losses.BinaryCrossentropy()})

    return full_model, attention_model
    