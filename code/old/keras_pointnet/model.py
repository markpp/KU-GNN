from tensorflow.keras.layers import Layer, Conv1D, MaxPooling1D, Flatten, Dropout, Input, BatchNormalization, Dense
from tensorflow.keras.layers import Reshape, Lambda, concatenate
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf

bn = True
do = False

# PointNet without input and feature transform sub networks
def Pointnet(input_points):
    g = Conv1D(64, 1, activation='relu')(input_points)
    g = BatchNormalization()(g)
    g = Conv1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(128, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(1024, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    global_feature = MaxPooling1D(pool_size=1024)(g)
    return global_feature

def Encoder(feature, n_latent, name="enc"):
    e = Dense(512, activation='relu')(feature)
    if bn:
        e = BatchNormalization()(e)
    if do:
        e = Dropout(0.5)(e)
    e = Dense(256, activation='relu')(e)
    if bn:
        e = BatchNormalization()(e)
    if do:
        e = Dropout(0.5)(e)
    e = Dense(n_latent, activation='linear')(e)
    z = Flatten(name='{}_z'.format(name))(e)
    return z

def Normal(latent):
    '''
    n = Dense(64, activation='relu', name='n0')(latent)
    if bn:
        n = BatchNormalization(name='nbn0')(n)
    if do:
        n = Dropout(0.5)(n)
    n = Dense(32, activation='relu', name='n1')(n)
    if bn:
        n = BatchNormalization(name='nbn1')(n)
    if do:
        n = Dropout(0.5)(n)
    n = Dense(16, activation='relu', name='n2')(n)
    '''
    n = Dense(48, activation='relu', name='n0')(latent)
    if bn:
        n = BatchNormalization(name='nbn0')(n)
    if do:
        n = Dropout(0.5)(n)
    n = Dense(24, activation='relu', name='n1')(n)
    if bn:
        n = BatchNormalization(name='nbn1')(n)
    if do:
        n = Dropout(0.5)(n)
    n = Dense(8, activation='relu', name='n2')(n)

    n = Dense(3, activation='linear', name='n3')(n)
    n = Flatten(name='norm')(n)
    return n


def Point2Pose(cloud_shape): #late split
    input_points = Input(shape=cloud_shape)
    # create PointNet model
    cloud_feature = Pointnet(input_points)

    z = Encoder(cloud_feature, 16)

    # normal regressor
    pred = Normal(z)
    model = Model(inputs=input_points, outputs=pred)
    return model
