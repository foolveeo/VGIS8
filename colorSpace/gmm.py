#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 18:59:36 2018

@author: fulvio
"""
from scipy import fftpack
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

def bgr_to_chR_chG(bgrImg):
    rows, cols, _ = bgrImg.shape
    
    chR = np.zeros((rows, cols), np.float)
    chG = np.zeros((rows, cols), np.float)
    chB = np.zeros((rows, cols), np.float)
    
    for x in range(0,rows):
        for y in range(0,cols):
            if  (bgrImg[x,y,1] != 0):
               
                r = float(bgrImg[x,y,2]) / float(255)
                g = float(bgrImg[x,y,1]) / float(255)
                b = float(bgrImg[x,y,0]) / float(255)
                
                # r chromaticity componen
                chR[x,y] = np.uint8(r*255 / (r+g+b))
                
                #g chromaticity component
                chG[x,y] = np.uint8(g*255 / (r+g+b))
                
                chB[x,y] = np.uint8(b*255 / (r+g+b))
                
    return chR, chB, chG




img = cv2.imread('RGB_40.png')
#hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
chR, chB, chG = bgr_to_chR_chG(img)

chromaticity = np.zeros((chR.shape[0], chR.shape[1], 3), np.uint8)
chromaticity[:,:,0] = chR
chromaticity[:,:,1] = chG
chromaticity[:,:,2] = chB

plt.figure()
plt.subplot(111) 
plt.imshow(chromaticity)
plt.title("RGB")


n_components = 4

X = np.zeros((chR.shape[0]*chB.shape[1],1))

X[:,0] = chG.ravel()
#X[:,1] = chB.ravel()
#X[:,2] = chR.ravel()

gmm = GaussianMixture(n_components).fit(X)
prob = gmm.predict_proba(X)
#plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
indexClasses = np.zeros(prob.shape[0])
probClasses = np.zeros(prob.shape[0])
for i in range(prob.shape[0]):
    probClasses[i] = np.amax(prob[i,:])
    if(probClasses[i] < 0.9):
        indexClasses[i] = -1
    else:
        indexClasses[i] = np.argmax(prob[i,:])
    
plt.figure()
plt.hist(probClasses,200,[0,1])
plt.title("probabilities")

probClasses = probClasses.reshape(chR.shape[0],chB.shape[1])
indexClasses = indexClasses.reshape(chR.shape[0],chB.shape[1])


plt.figure()
plt.imshow(probClasses, cmap='gray')
plt.title("probabilities")

plt.figure()
plt.imshow(indexClasses, cmap='nipy_spectral')
plt.title("classes")


cv2.imwrite("probabilities.png", probClasses)

#labels = labels.reshape((chR.shape[0],chB.shape[1]))
#plt.figure()
#labels = np.multiply(labels, 255/n_components)
#plt.imshow(labels, cmap='gray')

#chR = chR.ravel()
##chG = chG.ravel()
#chB = chB.ravel()
#

#
#r = cv2.selectROI(img[:,:,0])
##hCrop = h[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
##lCrop = l[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#chBCrop = chB[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#chRCrop = chR[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#chBCrop = chBCrop.ravel()
#chRCrop = chRCrop.ravel()
#
##


#chromaticity = np.zeros((chR.shape[0], 2), np.float)
#chromaticity[:,0] = chR
#chromaticity[:,1] = chB

#fft2 = fftpack.fft2(chB)
#plt.imshow(np.log10(abs(fft2)))
#plt.show()
#
###
##chromaticity = np.divide(chromaticity, 255)
##
#plt.scatter(chRCrop, chBCrop)
