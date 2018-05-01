# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:46:37 2018

@author: Fulvio Bertolini
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob


def getCameraExtrinsic(images, mtx, dist):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objp = np.multiply(objp, 0.0246)
    
   
   
    tvecs = []
    rvecs = [] 
    
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            
            _, rvec, tvec, inliers = cv2.solvePnPRansac(objp,corners2, mtx, dist)
            rvecs.append(rvec)
            tvecs.append(tvec)
            
            cv2.namedWindow("img", cv2.WINDOW_NORMAL )        # Create window with freedom of dimensions
            cv2.resizeWindow("img", 800, 600)              # Resize window to specified dimensions
    
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners,ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    
    
    return (rvecs, tvecs)


def getCameraIntrinsic(images):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objp = np.multiply(objp, 0.0246)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            
            cv2.namedWindow("img", cv2.WINDOW_NORMAL )        # Create window with freedom of dimensions
            cv2.resizeWindow("img", 800, 600)              # Resize window to specified dimensions
    
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners,ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
    
   
    
    return (mtx, dist, rvecs, tvecs)

#imagesZED = ["./UNO/samples/RGB_0.png",
#             "./DUE/samples/RGB_0.png",
#             "./TRE/samples/RGB_0.png",
#             "./QUATTRO/samples/RGB_0.png"]
#
imagesARKit = ["./uno.jpg",
               "./due.jpg",
               "./tre.jpg",
               "./quattro.jpg"]
#
#getCameraExtrinsic2(imagesZED)
#getCameraExtrinsic2(imagesARKit)
    
#used to compute intrinsic ZED
#images =  glob.glob('C:\\Users\\Fulvio Bertolini\\VGIS8\\cameraCalibration\\Calibration\\samples\\RGB_*.png')

#used to compute intrinsic iPhone
images_iPhone =  glob.glob('C:\\Users\\Fulvio Bertolini\\VGIS8\\cameraCalibration\\Calibration\\iOS\\IMG_*.jpg')


#(mtx, dist, rvecs, tvecs) = getCameraIntrinsic(images)

(mtxiPhone, distiPhone, _, _) = getCameraIntrinsic(images_iPhone)
rvecs, tvecs = getCameraExtrinsic(imagesARKit, mtxiPhone, distiPhone)
# used to get extrinsic ZED
images = ["./UNO/samples/RGB_0.png",
             "./DUE/samples/RGB_0.png",
             "./TRE/samples/RGB_0.png",
             "./QUATTRO/samples/RGB_0.png"]

# zed intrinsic parameters
cameraMatrixZED = np.zeros((3,3), np.float64)
cameraMatrixZED[0,0] = 660.445
cameraMatrixZED[0,2] = 650.905
cameraMatrixZED[1,1] = 660.579
cameraMatrixZED[1,2] = 327.352
cameraMatrixZED[2,2] = 1

distCoeffZED = np.zeros((1,5), np.float64)
distCoeffZED[0,0] = -0.00174692
distCoeffZED[0,1] = -0.0174969	
distCoeffZED[0,2] = -0.000182398
distCoeffZED[0,3] = -0.00751098
distCoeffZED[0,4] =	0.0219687




rvecs, tvecs = getCameraExtrinsic(images, cameraMatrix, distCoeff)


print("ah scemooooo")
