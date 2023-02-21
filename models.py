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

def build_LSTM(window_size,features,units):
    model = Sequential()
    model.add(InputLayer(input_shape=(window_size, features)))
    model.add(LSTM(units=units,return_sequences=True))
    model.add(LSTM(units=units))
    model.add(Dense(1, activation='softmax'))

    cp1 = ModelCheckpoint('model/', save_best_only=True)

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    return model


    