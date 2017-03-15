import os
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Input, merge
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, Reshape, Merge, MaxoutDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2, activity_l2
from keras import backend as K #enable tensorflow functions
from keras.layers.pooling import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from snapshot import SnapshotCallbackBuilder
import keras.metrics as metrics

PATH = os.path.dirname(os.path.abspath(__file__))

def train(run=0):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    #datagen = ImageDataGenerator(rotation_range=45,width_shift_range=0.2, height_shift_range=0.2)
    #datagen.fit(X_train)

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    nb_epoch = 80
    nb_musicians = 5
    snapshot = SnapshotCallbackBuilder(nb_epoch, nb_musicians, init_lr=0.006)
    #model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
    #                samples_per_epoch=len(X_train), nb_epoch=nb_epoch)
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

def dropconnect_lambda():
    pass

def inception_net(_input):
    x = Reshape((28, 28, 1))(_input)

    x = Convolution2D(16, 3, 3,subsample=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(48, 3, 3,subsample=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = mniny_inception_module(x, 1)
    x = mniny_inception_module(x, 2)
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
        predict = Dropout(0.25)(predict)
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

def evaluate_ensemble(Best=True):
    '''
    loads and evaluates an ensemle from the models in the model folder.
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(10000, 784)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_test = np_utils.to_categorical(y_test, 10)

    model_dirs = []
    for i in os.listdir('weights'):
        if '.h5' in i:
            if not Best:
                model_dirs.append(i)
            else:
                if 'Best' in i:
                    model_dirs.append(i)

    preds = []
    model = create_model()
    for mfile in model_dirs:
        print(os.path.join('weights',mfile))
        model.load_weights(os.path.join('weights',mfile))
        yPreds = model.predict(X_test, batch_size=128, verbose=1)
        preds.append(yPreds)

    weighted_predictions = np.zeros((X_test.shape[0], 10), dtype='float64')
    weight = 1./len(preds)
    for prediction in preds:
        weighted_predictions += weight * prediction
    y_pred =weighted_predictions

    Y_test = tf.convert_to_tensor(Y_test)
    y_pred = tf.convert_to_tensor(y_pred)

    loss = metrics.categorical_crossentropy(Y_test, y_pred)
    acc = metrics.categorical_accuracy(Y_test, y_pred)
    sess = tf.Session()
    print('--------------------------------------')
    print('ensemble')
    print('Test loss:', loss.eval(session=sess))
    print('error:', str((1.-acc.eval(session=sess))*100)+'%')
    print('--------------------------------------')

def evaluate(eval_all=False):
    '''
    evaluate models in the weights directory,
    defaults to only models with 'best'
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(10000, 784)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_test = np_utils.to_categorical(y_test, 10)
    evaluations = []

    for i in os.listdir('weights'):
        if '.h5' in i:
            if eval_all:
                evaluations.append(i)
            else:
                if 'Best' in i:
                    evaluations.append(i)
    print(evaluations)
    model = create_model()
    for run, i in enumerate(evaluations):
        model.load_weights(os.path.join('weights',i))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                    metrics=['categorical_accuracy'])
        score = model.evaluate(X_test, Y_test,
                            verbose=1)
        print('--------------------------------------')
        print('model'+str(run)+':')
        print('Test loss:', score[0])
        print('error:', str((1.-score[1])*100)+'%')

