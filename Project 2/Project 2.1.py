# -*- coding: utf-8 -*-
"""
author: Miguel Lopez
501035749
AER850 Project 2
"""
#%% Background and Introduction

"""
This project focuses on automating the visual inspection and detection of defects—such as cracks, missing screws, 
and paint degradation—using Deep Convolutional Neural Networks (DCNNs). 
These defects are critical to identify early to ensure the safety and structural integrity of aircraft. 
By using a dataset of defect images, the project aims to develop a DCNN model to classify these defects efficiently and accurately.
The project involves data processing, network design and architecture, performance evaluation and testing.

"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import matplotlib.pyplot as plt
from datetime import datetime
from numpy.random import seed

#%% Step 1 - Data Processing

seed(1)  # Seed for reproducibility

# Data Directories
data_dir = 'data'
train_data_dir = os.path.join(data_dir, 'train')
validation_data_dir = os.path.join(data_dir, 'valid')
saved_model_dir = 'saved_models'

# Dimensions of input images
img_width, img_height = 500, 500
input_shape = (img_width, img_height, 3)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 32

# Train and validation generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

#%% Step 2 - Neural Network Architecture Design
"""
The model consists of three convolutional layers, followed by max-pooling layers, a flatten layer, and two dense layers.
with the number of units and activation functions being tuned. Hyperparameter search and optimization is carried out using RandomSearch to optimize the validation accuracy.
"""
def build_model(hp):
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(hp.Int('conv1_filters', min_value=32, max_value=64, step=16),
                     (3, 3),
                     activation=hp.Choice('conv1_activation', values=['relu', 'leaky_relu']),
                     input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 2
    model.add(Conv2D(hp.Int('conv2_filters', min_value=32, max_value=64, step=32),
                     (3, 3),
                     activation=hp.Choice('conv2_activation', values=['relu', 'leaky_relu'])))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 3
    model.add(Conv2D(hp.Int('conv3_filters', min_value=32, max_value=64, step=32),
                     (3, 3),
                     activation=hp.Choice('conv3_activation', values=['relu', 'leaky_relu'])))
    model.add(MaxPooling2D((2, 2)))

    # Flatten layer to connect convolution layers to dense layers
    model.add(Flatten())

    # Dense Layer 1
    model.add(Dense(hp.Int('dense1_units', min_value=32, max_value=128, step=32),
                    activation=hp.Choice('dense1_activation', values=['relu', 'elu'])))

    # Dropout layer for regularization
    model.add(Dropout(0.5))

    # Dense Layer 2 (output layer)
    model.add(Dense(3, activation='softmax'))  # 3 classes for classification

    # Compile the model
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

#%% Step 3 Hyperparameter optimization

#The hyperparameter ranges and steps were chosen arbitrarely and optimized by using Keras RandomSearch
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=4,  # Limit to 4 trials to reduce training time, not all combinations were tested
    directory='my_dir',
    project_name='classification_with_tuning'
)

# Early stopping configuration to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Start the hyperparameter search
tuner.search(train_generator, epochs=5, validation_data=validation_generator, callbacks=[early_stopping])

# Retrieve the best model found by the tuner
best_model = tuner.get_best_models(num_models=1)[0]

# Summary of the best model
best_model.summary()

# Train the best model
epochs = 5
history = best_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator.filenames) // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator.filenames) // batch_size,
    callbacks=[early_stopping]
)

# Save the best model
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_path = os.path.join(saved_model_dir, f'tuned_model_{timestamp}.h5')
best_model.save(model_path)
print(f"Best model saved to: {model_path}")

#%% Step 4 - Model Evaluation
# Plotting Training and Validation Performance
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()