# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 09:39:49 2018

@author: Mike
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:58:48 2018

@author: Mike
"""

import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import backend as K

#if K.backend()=='tensorflow':
#    K.set_image_dim_ordering("th")

import Project_4 as p4
# This is the main function. You need to write the getModel and fitModel functions to pass to this.
# Call your functions 'myGetModel' and 'myFitModel'.
# The getModel function should accept an object of the CIFAR class, and return a compiled Keras CNN model. 
# In this function you will specify the network structure (including regularization) and the optimizer to 
# be used (and its parameters like learning rate), and run compile the model (in the Keras sense of running 
# model.compile).
# The fitModel function should accect two arguments. The first is the CNN model you return from your getModel 
# function, and the second is the CIFAR classed data object. It will return a trained Keras CNN model, which 
# will then be applied to the test data. In this function you will train the model, using the Keras model.fit 
# function. You will need to specify all parameters of the training algorithm (batch size, etc), and the 
# callbacks you will use (EarlyStopping and ModelCheckpoint). You will need to make sure you save and load 
# into the model the weight values of its best performing epoch.
def runImageClassification_Eval(getModel=None,fitModel=None,seed=7):
    # Fetch data. You may need to be connected to the internet the first time this is done.
    # After the first time, it should be available in your system. On the off chance this
    # is not the case on your system and you find yourself repeatedly downloading the data, 
    # you should change this code so you can load the data once and pass it to this function. 
    print("Preparing data...")
    data=p4.CIFAR(seed)
    x_test=data.x_test
    y_test=data.y_test
    data.x_test=None
    data.y_test=None
    
    # Create model 
    print("Creating model...")
    model=getModel(data)

    # Fit model
    print("Fitting model...")
    model=fitModel(model,data)

    # Evaluate on test data
    print("Evaluating model...")
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', score[1])

# Define Model
def get_model(input_dim,output_dim):

    # Define model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    ##opt = SGD(lr = 0.1, decay=1e-6, nesterov=True)
    opt = SGD(lr = 0.1, decay=1e-6)
    #opt=Adam(decay=1e-6)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def parGetModel (data):
    return get_model(data.input_dim,data.num_classes)

def parFitModel (model,data):
    # Fit model
    batch_size = 64 # 32 examples in a mini-batch, smaller batch size means more updates in one epoch

    # Create early stopping and checkpoint callbacks
    filepath="weights-best.hdf5"
    callbacks=[EarlyStopping(patience=20),
              ModelCheckpoint(filepath,monitor="val_acc",save_best_only=True,mode="max",verbose=1)]

    # Fit model    
    model.fit(data.x_train, data.y_train, batch_size=batch_size, epochs=100, 
              validation_data=(data.x_valid,data.y_valid),shuffle=True,callbacks=callbacks)

    # Load best weights
    model.load_weights(filepath)
    
    return (model)
