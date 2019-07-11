# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 20:29:28 2019

@author: xngu0004
"""

import numpy as np
import time
import pandas as pd
import tensorflow as tf
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import Input, Dense, GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, BatchNormalization, Activation
from keras.layers import LSTM, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping

##################################################
### To show elapse time
##################################################
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
##################################################
### GLOBAL VARIABLES
##################################################
COLUMN_NAMES = [
    'user',
    'activity',
    'timestamp',
    'x-axis',
    'y-axis',
    'z-axis'
]

LABELS = [
    'Downstairs',
    'Jogging',
    'Sitting',
    'Standing',
    'Upstairs',
    'Walking'
]

DATA_PATH = 'data/WISDM_ar_v1.1_raw.txt'

RANDOM_SEED = 13

# Data preprocessing
TIME_STEP = 100

# Model
N_CLASSES = 6
N_FEATURES = 3  # x-acceleration, y-acceleration, z-acceleration

# Hyperparameters
N_LSTM_LAYERS = 2
N_EPOCHS = 15
L2_LOSS = 0.0015
LEARNING_RATE = 0.0025

# Hyperparameters optimized
SEGMENT_TIME_SIZE = 180
N_HIDDEN_NEURONS = 30
BATCH_SIZE = 32

##################################################
### MAIN
##################################################
if __name__ == '__main__':

    # LOAD DATA
    data = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)
    data['z-axis'].replace({';': ''}, regex=True, inplace=True)
    data = data.dropna()

#    # SHOW ACTIVITY GRAPH
#    activity_type = data['activity'].value_counts().plot(kind='bar', title='Activity type')
#    plt.show()

    # DATA PREPROCESSING
    data_convoluted = []
    labels = []

    # Slide a "SEGMENT_TIME_SIZE" wide window with a step size of "TIME_STEP"
    for i in range(0, len(data) - SEGMENT_TIME_SIZE, TIME_STEP):
        x = data['x-axis'].values[i: i + SEGMENT_TIME_SIZE]
        y = data['y-axis'].values[i: i + SEGMENT_TIME_SIZE]
        z = data['z-axis'].values[i: i + SEGMENT_TIME_SIZE]
        data_convoluted.append([x, y, z])

        # Label for a data window is the label that appears most commonly
        label = stats.mode(data['activity'][i: i + SEGMENT_TIME_SIZE])[0][0]
        labels.append(label)

    # Convert to numpy
    data_convoluted = np.asarray(data_convoluted, dtype=np.float32).transpose(0, 2, 1)

    # One-hot encoding
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    print("Convoluted data shape: ", data_convoluted.shape)
    print("Labels shape:", labels.shape)

    # SPLIT INTO TRAINING AND TEST SETS
    X_train1, X_test, y_train1, y_test = train_test_split(data_convoluted, labels, test_size=0.2, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.25, random_state=RANDOM_SEED)
    print("X train size: ", len(X_train))
    print("X test size: ", len(X_test))
    print("X val size: ", len(X_val))
    print("y train size: ", len(y_train))
    print("y test size: ", len(y_test))
    print("y val size: ", len(y_val))

    ##### BUILD A MODEL
    #######-------- CNN -------------
    ip = Input(shape=(SEGMENT_TIME_SIZE, N_FEATURES), name='main_input')
    
    x = Conv1D(32, 7, padding='same', kernel_initializer='he_uniform')(ip)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(64, 5, padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    fea = GlobalAveragePooling1D()(x)

    out = Dense(N_CLASSES, kernel_regularizer=regularizers.l2(0.001), activation='softmax')(fea)

    model = Model(ip, out)
    #######-------- LSTM -------------
#    ip = Input(shape=(SEGMENT_TIME_SIZE, N_FEATURES), name='main_input')
#    
#    x = LSTM(64, return_sequences=True)(ip)
#    x = LSTM(64)(x)
#
#    out = Dense(N_CLASSES, kernel_regularizer=regularizers.l2(0.001), activation='softmax')(x)
#
#    #model.summary()
#    model = Model(ip, out)
    
    #######---------CNN+LSTM------------
#    ip = Input(shape=(SEGMENT_TIME_SIZE, N_FEATURES), name='main_input')
#    
#    x = Conv1D(16, 7, padding='same', kernel_initializer='he_uniform')(ip)
#    x = BatchNormalization()(x)
#    x = Activation('relu')(x)
#
#    x = Conv1D(32, 5, padding='same', kernel_initializer='he_uniform')(x)
#    x = BatchNormalization()(x)
#    x = Activation('relu')(x)
#    
#    x = LSTM(32, return_sequences=True)(x)
#    x = LSTM(32)(x)
#
#    out = Dense(N_CLASSES, kernel_regularizer=regularizers.l2(0.001), activation='softmax')(x)
#            
#    #model.summary()
#    model = Model(ip, out)
    
    #######-----------------------------
    
    ######### Training #########
    weight_fn = "./weights_simon.h5"
    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode='max', monitor='val_acc', save_best_only=True, save_weights_only=True)
    stop = EarlyStopping(monitor='val_loss', patience=15)
    time_callback = TimeHistory()
    callback_list = [model_checkpoint, time_callback, stop]
    optm = Adam(lr=LEARNING_RATE)

    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, callbacks = callback_list, verbose=2, validation_data=(X_val, y_val))

    model.load_weights(weight_fn)
    prediction = model.predict(X_test);
    p_pred = np.argmax(prediction, axis=1)
    p_test = np.argmax(y_test, axis=1)
    
    cr = classification_report(p_test, p_pred)
    cm = confusion_matrix(p_test, p_pred)
    
    print("-------------------- Result -----------------------")
    acc = np.sum(p_pred == p_test)/p_pred.shape[0]
    print('Accuracy: ' + str(acc))
    print(cr)
    print(cm)
    
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # CONFUSION MATRIX
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm/(np.sum(cm, axis=1, keepdims=1)), xticklabels=LABELS, yticklabels=LABELS, annot=True);
    plt.title("Confusion matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    print(time_callback.times)
