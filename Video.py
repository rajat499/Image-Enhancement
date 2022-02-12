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


file = './data/night_time_video_iitd.mp4'
file_rgb = './data/rgb_equalized_night_time_video_iitd.mp4'
file_hsv = './data/hsv_equalized_night_time_video_iitd.mp4'


# In[3]:


def convert_to_night_mode_rgb(image):
    ib, ig, ir = cv2.split(image)
    r, g, b = equalize_adapthist(np.array([ir, ig, ib]))
    image = np.dstack([b*255, g*255, r*255]).astype('uint8')
    return image

def convert_to_night_mode_hsv(image):
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    v = (equalize_adapthist(v)*255).astype('uint8')
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR).astype('uint8')


# In[4]:


def video_enhance(file, file_rgb, file_hsv):
    
    video = cv2.VideoCapture(file)
    
    frame = int(video.get(cv2.CAP_PROP_FPS))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    out_rgb = cv2.VideoWriter(file_rgb, cv2.VideoWriter_fourcc(*'MP4V'), frame, (w, h))
    out_hsv = cv2.VideoWriter(file_hsv, cv2.VideoWriter_fourcc(*'MP4V'), frame, (w, h))
    
    current = 0
    while True:
        ret, frame = video.read()
        if ret:
            out_rgb.write(convert_to_night_mode_rgb(frame))
            out_hsv.write(convert_to_night_mode_hsv(frame))
        else:
            break
        current += 1
        if current%30==0:
            print(current)
    out_rgb.release()
    out_hsv.release()
    video.release()


# In[5]:


video_enhance(file, file_rgb, file_hsv)


# In[ ]:




