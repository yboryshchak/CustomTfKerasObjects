import os
import math
import numpy as np
import os
import math
import numpy as np
from numpy import array, asarray, save
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Input, Dropout, GaussianNoise, concatenate
from tensorflow.python.keras.utils.vis_utils import plot_model
from functools import partial



def MyActivation(x):
    """ Custom Relu with keras beckend"""
    y = K.switch(x>0, 2*x, 0.1*x)
    return y

def ZoomTahn(x, Zoom==1):
    """ vertical stretch of tanh with keras beckend"""
    y =  Zoom*K.tanh(x)
    return y

@tf.custom_gradient
def NRootActivation(X, N=3, alpha=0.1**6):
    """ N-th root activation with |gradient|< alpha**(1 - 1/N) """
    power = 1/N
    dpow = 1 - power
    PX = -1* X
    if X>0:
        y = tf.math.pow(X, power)
        der = tf.math.powe(X+alpha, dpow)
    else:
        y = tf.math.pow(PX, power)
        der = tf.math.pow(PX+alpa, dpow)
    def NRoot_gradients(grad):
        return grad * der
    return y, NRoot_gradients


def exponential_decay(lr0, s, ebase=0.7 min_lr):
    """ exp decaying learnin rate with init value: lr0, base: ebase, exponent: epoch/s, and min value: min_lr """
    def exponential_decay_fn(epoch):
        clr = max(lr0 * ebase**(epoch / s), min_lr)
        return clr
    return exponential_decay_fn

def TwoDcoordinates(flatten_index, x_size):
    """ 2-D coordinates of flatten layer with horizontal size x_size """
    x_val = flatten_index % x_size
    y_val = flatten_index // x_size
    return [x_val, y_val]

def TwoDConnectList(x_size, y_size, distance):
    """ Outputs list of connections of 2-D flatten layer of size (x_size, y_size) within distanse from each neuron """
    connect_array = []
    Z = x_size * y_size
    for neuron in range(0, Z):
        x_val = TwoDcoordinates(neuron, x_size)[0]
        y_val = TwoDcoordinates(neuron, x_size)[1]
        Nconnect_list = []
        for neighbor in range(0, Z):
            x1_val = TwoDcoordinates(neighbor, x_size)[0]
            y1_val = TwoDcoordinates(neighbor, x_size)[1]
            dist_square = (x_val - x1_val)**2 + (y_val - y1_val)**2
            if  dist_square <= distance**2:
                Nconnect_list.append(neighbor)
        connect_array.append(Nconnect_list)
    return connect_array


def ConnectMatrix(PrevLayerSize, connect_list):
    """ Outputs initializer matrix for Custom Layer with the connection list = connect_list """
    vec_array = []
    for neuron in connect_list:
        conect_list = []
        for item in range(0, PrevLayerSize):
            if item in neuron:
                conect_list.append(1.0)
            else:
                conect_list.append(0.0)
        vec_array.append(conect_list)
    return vec_array

def SingleConnect(Size):
    """ Outputs initializer matrix for layer of size=Size, with one connection for each neuron """
    Lsize = int(Size)
    ConnecArray = []
    for neuron in range(Lsize):
        ConnectList = []
        for item in range(Lsize):
            if item==neuron:
                connection = 1.0
            else:
                connection = 0.0
            ConnectList.append(connection)
        ConnectArray.append(ConnectList)
    return ConnectArray


class SingleConnectCosLayer(tf.keras.layers.Layer):
    """ Layer with single connection and activation function cos(amplitude * x) """
    def __init__(self,  amplitude,  units):
        super(SingleConnectCosLayer, self).__init__()
        self.units = units
        self.amplitude = amplitude
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer=tf.constant_initializer(value=self.connect_matrix), trainable=False)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
    def call(self, inputs):
        out_lin = tf.matmul(inputs, self.w)
        out_ampl = tf.math.scalar_mul(self.amplitude, out_lin)
        cos_out = tf.math.cos(out_ampl)
        return tf.math.multiply(self.b, cos_out)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units}

class SingleConnectSinLayer(tf.keras.layers.Layer):
    """ Layer with single connection and activation function sin(amplitude * x) """
    def __init__(self,  amplitude,  units):
        super(SingleConnectSinLayer, self).__init__()
        self.units = units
        self.amplitude = amplitude
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer=tf.constant_initializer(value=self.connect_matrix), trainable=False)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
    def call(self, inputs):
        out_lin = tf.matmul(inputs, self.w)
        out_ampl = tf.math.scalar_mul(self.amplitude, out_lin)
        cos_out = tf.math.sin(out_ampl)
        return tf.math.multiply(self.b, cos_out)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units}


class SingleConnectFourierSeries(keras.models.Model):
    """ tf.keras.Model class that produce Fourier Sum for each dimension of the prediction function of order <= order using backprop """
    def __init__(self, units, order,  **kwargs):
        super().__init__(**kwargs)
        self.order = int(order)
        self.units = units
        self.inputlayer = tf.keras.layers.InputLayer()
        self.cos = [SingleConnectCosLayer(amplitude, units) for amplitude in range(0, order)]
        self.sin = [SingleConnectSinLayer(amplitude, units) for amplitude in range(0, order)]
        self.summ = tf.keras.layers.Add()
        
    def call(self, inputs):
        Z = self.inputlayer(inputs)
        cos_array = self.summ([layer(Z) for layer in self.cos])
        sin_array = self.summ([layer(Z) for layer in self.sin])
        return self.summ([cos_array, sin_array])
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units, "order": self.order}




class CustomLayer(tf.keras.layers.Layer):
    """ Custom layer with initialize matrix = connect_matrix. """
    def __init__(self,  activation, units, connect_matrix):
        super(CustomLayer, self).__init__()
        self.units = units
        self.connect_matrix = np.asarray(connect_matrix, dtype='float32')
        self.activation = tf.keras.activations.get(activation)
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer=tf.constant_initializer(value=self.connect_matrix), trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
        self.m = self.add_weight(shape=(input_shape[-1], self.units), initializer=tf.constant_initializer(value=self.connect_matrix), trainable=False)
        #super().build(batch_input_shape)
    def call(self, inputs):
        wm = tf.math.multiply(self.w, self.m)
        out_lin = tf.matmul(inputs, wm) + self.b
        return self.activation(out_lin)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units, "activation": keras.activations.serialize(self.activation)}


class CosLayer(tf.keras.layers.Layer):
    " Custom dense layer with cos(amplitude * x) activation function """
    def __init__(self,  amplitude,  units):
        super(CosLayer, self).__init__()
        self.units = units
        self.amplitude = amplitude
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='he_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
    def call(self, inputs):
        out_lin = tf.matmul(inputs, self.w)
        out_ampl = tf.math.scalar_mul(self.amplitude, out_lin)
        cos_out = tf.math.cos(out_ampl)
        return tf.math.multiply(self.b, cos_out)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units}

class SinLayer(tf.keras.layers.Layer):
    " Custom dense layer with sin(amplitude * x) activation function """
    def __init__(self,  amplitude,  units):
        super(SinLayer, self).__init__()
        self.units = units
        self.amplitude = amplitude
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='he_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
    def call(self, inputs):
        out_lin = tf.matmul(inputs, self.w)
        out_ampl = tf.math.scalar_mul(self.amplitude, out_lin)
        cos_out = tf.math.sin(out_ampl)
        return tf.math.multiply(self.b, cos_out)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units}

class FourierSeries(keras.models.Model):
    """ tf.keras.Model class that produce Fourier Sum of dimension= units for prediction function of order <= order """
    def __init__(self, units, order,  **kwargs):
        super().__init__(**kwargs)
        self.order = int(order)
        self.units = units
        self.flattlayer = tf.keras.layers.Flatten()
        self.cos = [CosLayer(amplitude, units) for amplitude in range(0, order)]
        self.sin = [SinLayer(amplitude, units) for amplitude in range(0, order)]
        self.summ = tf.keras.layers.Add()
        
    def call(self, inputs):
        Z = self.flattlayer(inputs)
        cos_array = self.summ([layer(Z) for layer in self.cos])
        sin_array = self.summ([layer(Z) for layer in self.sin])
        return self.summ([cos_array, sin_array])
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units, "order": self.order}



