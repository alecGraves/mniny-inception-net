import os
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Input, merge
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, Reshape, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2, activity_l2
from keras import backend as K #enable tensorflow functions
from keras.layers.pooling import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from snapshot import SnapshotCallbackBuilder


PATH = os.path.dirname(os.path.abspath(__file__))

def train(run=0):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    nb_epoch = 80
    nb_musicians = 5
    snapshot = SnapshotCallbackBuilder(nb_epoch, nb_musicians, init_lr=0.006)

    model.fit(X_train, Y_train,
              batch_size=1024, nb_epoch=nb_epoch,
              verbose=1,
              validation_data=(X_test, Y_test),
              callbacks=snapshot.get_callbacks('snap-model'+str(run)))

    model = load_model("weights/%s-Best.h5" % ('snap-model'+str(run)))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy'])
    score = model.evaluate(X_test, Y_test,
                           verbose=0)
    print('--------------------------------------')
    print('model'+str(run)+':')
    print('Test loss:', score[0])
    print('error:', str((1.-score[1])*100)+'%')
    return score

def create_model():
    _input = Input((784,))
    incep1 = inception_net(_input)
    out = incep1
    model = Model(input=_input, output=[out])
    return model


def conv_net(_input):
    x = Reshape((28, 28, 1,))(_input)
    x = Convolution2D(32, 3, 3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(32, 3, 3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10)(x)
    out = Activation('softmax')(x)
    return out


def fire_net_xs(_input):
    '''
	note: fire_net_xs is larger now due to giant 5x5 convolution in first layer
	'''
    x = Reshape((28, 28, 1,))(_input)
    x = Convolution2D(32, 3, 3, subsample=(1, 1), border_mode='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, fire_id=2, squeeze=8, expand=32)
    x = fire_module(x, fire_id=3, squeeze=8, expand=32)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, fire_id=3, squeeze=8, expand=32)
    x = fire_module(x, fire_id=6, squeeze=16, expand=64)
    x = fire_module(x, fire_id=7, squeeze=16, expand=64)
    x = Dropout(0.4)(x)

    x = Convolution2D(10, 1, 1, border_mode='valid')(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax')(x)
    return out

def fire_net(_input):
    x = Reshape((28, 28, 1,))(_input)
    x = Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)


    x = fire_module(x, fire_id=6, squeeze=32, expand=128)
    x = fire_module(x, fire_id=7, squeeze=32, expand=128)
    x = Dropout(0.6)(x)

    x = Convolution2D(10, 1, 1, border_mode='valid')(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax')(x)
    return out

def fire_module(x, fire_id, squeeze=16, expand=64):
    x = Convolution2D(squeeze, 1, 1, border_mode='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    left = Convolution2D(expand, 1, 1, border_mode='valid')(x)
    left = BatchNormalization()(left)
    left = Activation('relu')(left)

    right = Convolution2D(expand, 3, 3, border_mode='same')(x)
    right = BatchNormalization()(right)
    right = Activation('relu')(right)

    x = merge([left, right], mode='concat', concat_axis=3)
    return x
def dropconnect_lambda():
    pass

def inception_net(_input):
    x = Reshape((28, 28, 1))(_input)
    #x = Convolution2D(32, 3, 3, subsample=(1, 1))(x)
    #x = Activation('relu')(x)
    x = Convolution2D(16, 3, 3,subsample=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(48, 3, 3,subsample=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2,2))(x)
    x = mniny_inception_module(x, 1)
    x = mniny_inception_module(x, 2)
    #x = MaxPooling2D((3, 3), strides=(2,2))(x)
    x = mniny_inception_module(x, 2)
    x, soft1 = mniny_inception_module(x, 3, True)
    x = mniny_inception_module(x, 3)
    x = mniny_inception_module(x, 3)
    x, soft2 = mniny_inception_module(x, 4, True)
    x = MaxPooling2D((3, 3), strides=(2,2))(x)
    x = mniny_inception_module(x, 4)
    x = mniny_inception_module(x, 5)
    x = AveragePooling2D((5, 5), strides=(1, 1))(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    soft3 = Dense(10, activation='softmax')(x)
    out = Merge(mode='ave', concat_axis=1)([soft1, soft2, soft3])
    return out

def mniny_inception_module(x, scale=1, predict=False):
    '''
    x is input layer, scale is factor to scale kernel sizes by
    '''
    x11 = Convolution2D(int(16*scale), 1, 1, border_mode='valid')(x)
    x11 = BatchNormalization()(x11)
    x11 = Activation('relu')(x11)

    x33 = Convolution2D(int(24*scale), 1, 1)(x)
    x33 = BatchNormalization()(x33)
    x33 = Activation('relu')(x33)
    x33 = Convolution2D(int(32*scale), 3, 3, border_mode='same')(x33)
    x33 = BatchNormalization()(x33)
    x33 = Activation('relu')(x33)

    x55 = Convolution2D(int(4*scale), 1, 1)(x)
    x55 = BatchNormalization()(x55)
    x55 = Activation('relu')(x55)
    x55 = Convolution2D(int(8*scale), 5, 5, border_mode='same')(x55)
    x55 = BatchNormalization()(x55)
    x55 = Activation('relu')(x55)

    x33p = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
    x33p = Convolution2D(int(8*scale), 1, 1)(x33p)
    x33p = BatchNormalization()(x33p)
    x33p = Activation('relu')(x33p)

    out = merge([x11, x33, x55, x33p], mode='concat', concat_axis=3)

    if predict:
        predict = AveragePooling2D((5, 5), strides=(1, 1))(x)
        predict = Convolution2D(int(8*scale), 1, 1)(predict)
        predict = BatchNormalization()(predict)
        predict = Activation('relu')(predict)
        predict = Dropout(0.35)(predict)
        predict = Flatten()(predict)
        predict = Dense(120)(predict)
        predict = BatchNormalization()(predict)
        predict = Activation('relu')(predict)
        predict = Dense(10, activation='softmax')(predict)
        return out, predict

    return out

def test_model():
    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print("MODEL COMPILES SUCCESSFULLY")

#test_model()

run = 0
while True:
    train(run)
    run += 1


# Performance history (notable cases):
#Ensemble (2 fire_net, 3 conv_net):
#Epoch 30/30
#60000/60000 [==============================] - 33s - loss: 0.0053 - val_loss: 0.0229
#10000/10000 [==============================] - 8s
#Test score: 0.0229417834882
