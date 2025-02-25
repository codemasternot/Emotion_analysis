#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# In[4]:


train_dir = "C:\\Users\\Stephen\\archivemood\\train"
test_dir = "C:\\Users\\Stephen\\archivemood\\test"


# In[5]:


img_size = (48, 48)  # FER2013 image size
batch_size = 64


# In[6]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)


# In[7]:


test_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation for test

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)


# In[8]:


test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)


# In[9]:


# CNN Model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # Number of emotion classes
])


# In[10]:


# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[20]:


# Train model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=75,
    verbose=1
)


# In[21]:


model.save("myemotion_cnn_model.h5")


# In[ ]:





# In[ ]:




