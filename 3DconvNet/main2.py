# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 21:54:23 2019
@author: xngu0004
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Input, BatchNormalization, GlobalAveragePooling3D, Conv3D
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# image specification
img_rows,img_cols,img_depth = 20, 20, 50

# Training data
X_tr = []           # variable to store entire dataset X
label = []          # variable to store entire dataset y
split_no = 4        # number of augmented segments per video

#Reading boxing action class
activities = ['boxing','handclapping','handwaving','jogging','running','walking']

for act in activities:
    listing = os.listdir('./'+str(act))
    ipy = activities.index(act)

    for vid in listing:
        vid = './'+str(act) + '/' + vid
        frames1 = []
        frames2 = []
        frames3 = []
        frames4 = []
        cap = cv2.VideoCapture(vid)
        fps = cap.get(5) # Frame rate
        #print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
  
        for k in range(split_no*img_depth):
            ret, frame = cap.read()
            frame = cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (k < img_depth):  
                frames1.append(gray)
            elif (k < 2*img_depth):  
                frames2.append(gray)
            elif (k < 3*img_depth):  
                frames3.append(gray)
            elif (k < 4*img_depth):  
                frames4.append(gray)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        inputt1 = np.array(frames1)
        inputt2 = np.array(frames2)
        inputt3 = np.array(frames3)
        inputt4 = np.array(frames4)
        ipt1 = np.rollaxis(np.rollaxis(inputt1,2,0),2,0)
        ipt2 = np.rollaxis(np.rollaxis(inputt2,2,0),2,0)
        ipt3 = np.rollaxis(np.rollaxis(inputt3,2,0),2,0)
        ipt4 = np.rollaxis(np.rollaxis(inputt4,2,0),2,0)
        X_tr.append(ipt1)
        X_tr.append(ipt2)
        X_tr.append(ipt3)
        X_tr.append(ipt4)
        for k in range(split_no):
            label.append(ipy)
        
    X_tr_1y = np.array(X_tr)
    print (len(X_tr_1y))

X_tr_array = np.array(X_tr)   # convert the frames read into array
num_samples = len(X_tr_array) 

train_data = [X_tr_array,label]
(X_train, y_train) = (train_data[0],train_data[1])
print('X_Train shape:', X_train.shape)

train_set = np.zeros((num_samples, 1, img_rows,img_cols,img_depth))

for h in range(num_samples):
    train_set[h][0][:][:][:] = X_train[h,:,:,:]
  
print(train_set.shape, 'train samples')

# CNN Training parameters
batch_size = 2
nb_epoch = 24
nb_classes = len(activities)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)

# Pre-processing
train_set = train_set.astype('float32')
train_set -= np.mean(train_set)
train_set /= np.max(train_set)

# Split the data
X_train_new1,X_test_new,y_train_new1,y_test_new = train_test_split(train_set, Y_train, test_size=0.2, random_state=4)
X_train_new,X_val_new,y_train_new,y_val_new = train_test_split(X_train_new1, y_train_new1, test_size=0.25, random_state=5)

# Define model
inputs = Input(shape=(1,img_rows, img_cols, img_depth))

conv1 = Conv3D(8, (5, 5, 5), activation = 'relu', padding='same')(inputs)
conv1 = BatchNormalization(axis = 1)(conv1)
conv1 = Conv3D(16, (5, 5, 5), activation = 'relu', padding='same')(conv1)
conv1 = BatchNormalization(axis = 1)(conv1)
pool1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)

conv2 = Conv3D(16, (3, 3, 3), activation = 'relu', padding='same')(pool1)
conv2 = BatchNormalization(axis = 1)(conv2)
conv2 = Conv3D(32, (3, 3, 3), activation = 'relu', padding='same')(conv2)
conv2 = BatchNormalization(axis = 1)(conv2)
pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)

conv3 = Conv3D(32, (3, 3, 3), activation = 'relu', padding='same')(pool2)
conv3 = BatchNormalization(axis = 1)(conv3)
conv3 = Conv3D(64, (3, 3, 3), activation = 'relu', padding='same')(conv3)
conv3 = BatchNormalization(axis = 1)(conv3)
pool3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)

output = GlobalAveragePooling3D()(pool3)
output = Dense(nb_classes, activation='softmax', name = 'predictions')(output)

model = Model(inputs, output)
model.summary()

weight_fn = "./weights_action.h5"
model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode='max', monitor='val_acc', save_best_only=True, save_weights_only=True)
callback_list = [model_checkpoint]
optm = Adam(lr = 0.0025)
model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
hist = model.fit(X_train_new, y_train_new, 
                 validation_data=(X_val_new,y_val_new), 
                 batch_size=batch_size,
                 epochs = nb_epoch,
                 callbacks = callback_list)

# Evaluate the model
model.load_weights(weight_fn)
prediction = model.predict(X_test_new);
p_pred = np.argmax(prediction, axis = 1)
p_test = np.argmax(y_test_new, axis = 1)
    
cr = classification_report(p_test, p_pred)
cm = confusion_matrix(p_test, p_pred)

print("-------------------- Result -----------------------")
acc = np.sum(p_pred == p_test)/p_pred.shape[0]
print('Accuracy: ' + str(acc))
print(cr)
print(cm)

# Plot the results
# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
    
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
################################################################