# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:46:37 2018

@author: Fulvio Bertolini
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp = np.multiply(objp, 0.0246)

# Arrays to store object points and image points from all the images.
objpointsZed = [] # 3d point in real world space
imgpointsZed = [] # 2d points in image plane.

objpointsIphone = [] # 3d point in real world space
imgpointsIphone = [] # 2d points in image plane.

imagesZed = ["C:\\Users\\Fulvio Bertolini\\Documents\\Python\\cameraCalibration\\x7cm\\samples\\RGB_0.png"]


imagesIphone = ["C:\\Users\\Fulvio Bertolini\\OneDrive - Aalborg Universitet\\IMG_1971.jpg"]



for fname in imagesZed:
    imgZed = cv2.imread(fname)
    imgmatpotlib = imgZed;
    grayZed = cv2.cvtColor(imgZed,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retZed, cornersZed = cv2.findChessboardCorners(grayZed, (9,6),None)

    # If found, add object points, image points (after refining them)
    if retZed == True:
        objpointsZed.append(objp)

        cv2.cornerSubPix(grayZed,cornersZed,(11,11),(-1,-1),criteria)
        imgpointsZed.append(cornersZed)
        
        cv2.namedWindow("img", cv2.WINDOW_NORMAL )        # Create window with freedom of dimensions
        cv2.resizeWindow("img", 800, 600)              # Resize window to specified dimensions

        # Draw and display the corners
        cv2.drawChessboardCorners(imgZed, (9,6), cornersZed,retZed)
        cv2.imshow('img',imgZed)
        cv2.waitKey(5000)

cv2.destroyAllWindows()


retZed, mtxZed, distZed, rvecs, tvecs = cv2.calibrateCamera(objpointsZed, imgpointsZed, grayZed.shape[::-1],None,None)


rvecsZedPnp, tvecsZedPnp, _ = cv2.solvePnPRansac(objpointsZed[0], cornersZed, mtxZed, distZed )
rotMZed = cv2.Rodrigues(rvecsZed[0])

zedMatrix = np.zeros((4,4), np.float32)
zedMatrix[3,3] = 1

zedMatrix[0:3, 0:3] = rotMZed[0]
zedMatrix[0:3, 3] = tvecsZed[0][0:3,0]

print("zed tvec 0:\n", tvecsZed[0])
print("zed tvec 7cm x :\n", tvecsZed[1])


for fname in imagesIphone:
    imgIphone = cv2.imread(fname)
    imgmatpotlib = imgIphone;
    grayIphone = cv2.cvtColor(imgIphone,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retIphone, cornersIphone = cv2.findChessboardCorners(grayIphone, (9,6),None)
    
    # If found, add object points, image points (after refining them)
    if retIphone == True:
        objpointsIphone.append(objp)

        cv2.cornerSubPix(grayIphone,cornersIphone,(11,11),(-1,-1),criteria)
        imgpointsIphone.append(cornersIphone)
        
        cv2.namedWindow("img", cv2.WINDOW_NORMAL )        # Create window with freedom of dimensions
        cv2.resizeWindow("img", 800, 600)              # Resize window to specified dimensions

        # Draw and display the corners
        cv2.drawChessboardCorners(imgIphone, (9,6), cornersIphone,retIphone)
        cv2.imshow('img',imgIphone)
        cv2.waitKey(5000)

cv2.destroyAllWindows()


retIphone, mtxIphone, distIphone, rvecsIphone, tvecsIphone = cv2.calibrateCamera(objpointsIphone, imgpointsIphone, grayIphone.shape[::-1],None,None)

rotMIphone = cv2.Rodrigues(rvecsIphone[0])

iphoneMatrix = np.zeros((4,4), np.float32)
iphoneMatrix[3,3] = 1

iphoneMatrix[0:3, 0:3] = rotMIphone[0]
iphoneMatrix[0:3, 3] = tvecsIphone[0][0:3,0]

print("iphone tvec :\n", tvecsIphone[0])
