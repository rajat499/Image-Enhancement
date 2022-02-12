#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage.exposure import equalize_adapthist
import cv2
import numpy as np
from utils import read_image
import os
from PIL import Image


# In[2]:


night_off = 'night_mode_off.jpg'
night_on = 'night_mode_on.jpg'
fog = 'foggy.png'
clear = 'clear.png'


# In[3]:


def convert_to_night_mode_rgb(folder, night=True):
    night_image = read_image(os.path.join(folder, night_off if night else fog))
    ir, ig, ib = cv2.split(night_image)
    r, g, b = equalize_adapthist(np.array([ir, ig, ib]))
    image = Image.fromarray(np.dstack([r*255, g*255, b*255]).astype('uint8'))
    path = os.path.join(folder, 'equalized_rgb_transformed.png')
    image.save(path)


# In[4]:


def convert_to_night_mode_hsv(folder, night=True):
    night_image = cv2.imread(os.path.join(folder, night_off if night else fog), 1)
    h, s, v = cv2.split(cv2.cvtColor(night_image, cv2.COLOR_BGR2HSV))
    v = (equalize_adapthist(v)*255).astype('uint8')
    r, g, b = cv2.split(cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB))
    image = Image.fromarray(np.dstack([r, g, b]).astype('uint8'))
    path = os.path.join(folder, 'equalized_hsv_transformed.png')
    image.save(path)


# In[5]:


folder = './data/night'
night = True
for i in range(1, 5):
    path = os.path.join(folder, str(i))
    convert_to_night_mode_rgb(path, night)
    convert_to_night_mode_hsv(path, night)


# In[6]:


folder = './data/fog'
night = False
for i in range(1, 4):
    path = os.path.join(folder, str(i))
    convert_to_night_mode_rgb(path, night)
    convert_to_night_mode_hsv(path, night)


# In[ ]:




