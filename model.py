import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, Input, Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

def build_model(input_length = 4 * 14, alpha = 0.001):
    input_layer = Input(shape=(input_length,6)) # 6 fields in json
    flatten = Flatten()(input_layer)
    layer_1 = Dense(input_length * 3)(flatten)
    layer_2 = Dense(input_length * 3 // 2)(layer_1)
    layer_3 = Dense(input_length * 3 // 4)(layer_2)
    output_layer = Dense(4, activation='softmax')(layer_3)
    model = Model(input_layer, output_layer)
    model.compile(Adam(), 'mse', metrics=['accuracy'])

    return model
