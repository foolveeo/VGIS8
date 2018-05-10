# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:47:12 2018

@author: Fulvio Bertolini
"""

import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    
    h = frame[:, :, 0]
    l = frame[:, :, 1]
    s = frame[:, :, 2]
    
    cv2.imshow('Hue',h)
    cv2.imshow('Saturation',s)
    cv2.imshow('Light',l)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()