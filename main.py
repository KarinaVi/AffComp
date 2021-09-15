# loss: 1.2922 - accuracy: 0.5135 - val_loss: 1.0811 - val_accuracy: 0.5883

import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os


from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Part I: Model Creation
# defining some variables
num_classes = 7  # number of emotions (classes)
img_rows, img_cols = 48, 48
batch_size = 32

# load data in variables
train_data_dir = 'FER2013/train'
test_data_dir = 'FER2013/test'

# use Image Augmentation techniques to expand the size of the training dataset
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    color_mode='grayscale',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

model = Sequential()

# Block-1
'''
Conv2D layer: creates a conv layer for the network mit 32 filters and a filter size of (3,3) 
              with padding=’same’ to pad the image and using the kernel initializer he_normal
Activation layer: ELU
BatchNormalization: normalize the activations of the previous layer at each batch
MaxPooling2D: Downsamples the input representation by taking the maximum value over the window 
              defined by pool_size for each dimension along the features axis
              pool_size is (2,2)
Dropout: 0.5 - randomly ignore half of the neurons
'''
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block-2
'''
Same as Block-1, but Conv2D layer has now 64 filters
'''
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block-3
'''
Same as Block-1, but Conv2D layer has now 128 filters
'''
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block-4
'''
Same as Block-1, but Conv2D layer has now 256 filters
'''
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block-5
'''
Flatten layer: flatten the output of the prev layers in a flat layer/form of a vector
Dense layer: densely connected layer where each neuron is connected to every other neuron
             64 neurons with a kernel initializer (he_normal)
followed by activation with ELU, bacth normalization and 50% dropout
'''
model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6
'''
same as Block-5 but without flatten layer
'''
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7
'''
Dense layer: num_classes creates a dense layer having units=number of classes with he_normal initializer
activation-lay: softmax layer used fo mulri-class classifications
'''
model.add(Dense(num_classes, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

'''
Checkpoint monitors the validation loss and tries to minimize the loss using mode='min'
When checkpoint is reached it will save the best trained weights
verbose=1 is for visualization when the checkpoint is created
'''
checkpoint = ModelCheckpoint('EmotionDetectionModel',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

'''
earlystop stops the execution early by checking the properties
monitor: validation loss
min_delta: minimum change in the monitored quantity
patience: number of epochs with no improvement after which training will be stopped
restore_best_weights: whether to restore model weights from the epoche with the best value of the monitored quantity
'''
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True)

'''
reduce_lr monitors the validation loss and if no improvement is seen for the patience number of epochs,
the learning rate is reduced
factor: factor by which the learning rate will be reduced
min_delta: threshold for measuring new optimum
'''
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop, checkpoint, reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0015),   # Adam Optimizer and lr=0.001
              metrics=['accuracy'])

nb_train_samples = 28709
nb_test_samples = 7178
epochs = 25

# Fit the model
history = model.fit(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=test_generator,
                    validation_steps=nb_test_samples // batch_size)

# list all data in history
# print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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





