# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:46:37 2018

@author: Fulvio Bertolini
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
  
def multiplyMatrices4x4(rvec_iPhone, tvec_iPhone, rvec_ZED, tvec_ZED):
    
    
    rvec_iPhone[1,0] = rvec_iPhone[1,0] * -1
    rvec_ZED[1,0] = rvec_ZED[1,0] * -1
    
    rotM_iPhone = cv2.Rodrigues(rvec_iPhone)[0]
    rotM_ZED = cv2.Rodrigues(rvec_ZED)[0]
    
    rotM_ZED_inverse = np.linalg.inv(rotM_ZED)
    
    iPhone_2_ChB = np.zeros((4,4), np.float)
    ZED_2_ChB = np.zeros((4,4), np.float)
    
    iPhone_2_ChB[0:3,0:3] = rotM_iPhone
    ZED_2_ChB[0:3,0:3] = rotM_ZED
    
    iPhone_2_ChB[0,3] = tvec_iPhone[0]
    iPhone_2_ChB[1,3] = tvec_iPhone[1]
    iPhone_2_ChB[2,3] = tvec_iPhone[2]
    ZED_2_ChB[0,3] = tvec_ZED[0]
    ZED_2_ChB[1,3] = tvec_ZED[1]
    ZED_2_ChB[2,3] = tvec_ZED[2]
    
    iPhone_2_ChB[3,3] = 1
    ZED_2_ChB[3,3] = 1
    
    ChB_2_ZED = np.linalg.inv(ZED_2_ChB)
    
    iPhone_2_ZED = np.matmul(iPhone_2_ChB, ChB_2_ZED)
    
    
    tvec = iPhone_2_ZED[0:3,3]
    rvec = cv2.Rodrigues(iPhone_2_ZED[0:3,0:3])[0]

    return rvec, tvec
    
#
#def transposeIPHONE(rvec_iPhone, tvec_iPhone, rvec_ZED, tvec_ZED):
#    
#def transposeZED(rvec_iPhone, tvec_iPhone, rvec_ZED, tvec_ZED):
#    
#    
#def transposeBoth(rvec_iPhone, tvec_iPhone, rvec_ZED, tvec_ZED):
#    
#    

def computeARKit_2_ZED_matrix(rvec_iPhone, tvec_iPhone, rvec_ZED, tvec_ZED):

    rvec, tvec = multiplyMatrices4x4(rvec_iPhone, tvec_iPhone, rvec_ZED, tvec_ZED)
    
    
    return rvec, tvec
    
    

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
            
            #cv2.namedWindow("img", cv2.WINDOW_NORMAL )        # Create window with freedom of dimensions
            #cv2.resizeWindow("img", 800, 600)              # Resize window to specified dimensions
    
            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (9,6), corners,ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(500)
    
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
            
            #cv2.namedWindow("img", cv2.WINDOW_NORMAL )        # Create window with freedom of dimensions
            #cv2.resizeWindow("img", 800, 600)              # Resize window to specified dimensions
    
            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (9,6), corners,ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(500)
    
    #cv2.destroyAllWindows()
    
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
    
   
    
    return (mtx, dist, rvecs, tvecs)


# zed intrinsic parameters
cameraMatrix_ZED = np.zeros((3,3), np.float64)
cameraMatrix_ZED[0,0] = 660.445
cameraMatrix_ZED[0,2] = 650.905
cameraMatrix_ZED[1,1] = 660.579
cameraMatrix_ZED[1,2] = 327.352
cameraMatrix_ZED[2,2] = 1

distCoeff_ZED = np.zeros((1,5), np.float64)
distCoeff_ZED[0,0] = -0.00174692
distCoeff_ZED[0,1] = -0.0174969	
distCoeff_ZED[0,2] = -0.000182398
distCoeff_ZED[0,3] = -0.00751098
distCoeff_ZED[0,4] =  0.0219687

cameraMatrix_iPhone = np.zeros((3,3), np.float64)
cameraMatrix_iPhone[0,0] = 3513.23
cameraMatrix_iPhone[0,2] = 1542.31
cameraMatrix_iPhone[1,1] = 3519.98
cameraMatrix_iPhone[1,2] = 2089.89
cameraMatrix_iPhone[2,2] = 1

distCoeff_iPhone = np.zeros((1,5), np.float64)
distCoeff_iPhone[0,0] = 0.293461	
distCoeff_iPhone[0,1] = -1.3054	
distCoeff_iPhone[0,2] = 0.0132138	
distCoeff_iPhone[0,3] = 0.00104743
distCoeff_iPhone[0,4] = 2.50754


images_ZED = ["./3Maggio/samples/RGB_0.png"]
images_iPhone = ["./3Maggio/samples/IMG_2028.JPG"]

rvecs_ZED, tvecs_ZED = getCameraExtrinsic(images_ZED, cameraMatrix_ZED, distCoeff_ZED)
rvecs_iPhone, tvecs_iPhone = getCameraExtrinsic(images_iPhone, cameraMatrix_iPhone, distCoeff_iPhone)



eulerAngles_ARKit_2_ZED, translation_ARKit_2_ZED = computeARKit_2_ZED_matrix(rvecs_iPhone[0], tvecs_iPhone[0], rvecs_ZED[0], tvecs_ZED[0])
eulerAngles_ARKit_2_ZED_degrees = np.multiply(180/np.pi, eulerAngles_ARKit_2_ZED)
print ("euler angles:\n",eulerAngles_ARKit_2_ZED)
print("translation:\n", translation_ARKit_2_ZED)
eulerAngleFile = open("../TCP/eulerAngles.txt", "a+")
eulerAngleFile.write(str(eulerAngles_ARKit_2_ZED_degrees[0,0]) + " " + str(eulerAngles_ARKit_2_ZED_degrees[1,0]) + " " + str(eulerAngles_ARKit_2_ZED_degrees[2,0]) + "\n")
eulerAngleFile.close()

print("ah scemooooo")




