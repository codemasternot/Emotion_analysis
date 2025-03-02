#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


test_dir = "C:\\Users\\Stephen\\archivemood\\test"


# In[4]:


model = load_model("C:\\Users\\Stephen\\myemotion_cnn_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# In[5]:


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    shuffle=False  
)


# In[6]:


y_true = test_generator.classes

y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)  

cm = confusion_matrix(y_true, y_pred)


# In[7]:


plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Emotion Detection")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




