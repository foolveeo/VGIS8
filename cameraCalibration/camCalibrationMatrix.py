# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:46:37 2018

@author: Fulvio Bertolini
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob



def computeARKit_2_ZED_matrix(rvec_iPhone, tvec_iPhone, rvec_ZED, tvec_ZED):
    ## iphone -> checkerboard
    ## checkerboard .ì-> zed
    
    rvec_iPhone[1] = rvec_iPhone[1] * -1
    rvec_ZED[1] = rvec_ZED[1] * -1

    tvec_iPhone[1] = tvec_iPhone[1] * -1
    tvec_ZED[1] = tvec_ZED[1] * -1

    rotM_iPhone = cv2.Rodrigues(rvec_iPhone)[0]
    rotM_iPhone = np.transpose(rotM_iPhone)
    rotM_ZED = cv2.Rodrigues(rvec_ZED)[0]
    
    matrix_iPhone = np.zeros((4,4), np.float)
    matrix_ZED = np.zeros((4,4), np.float)
    
    matrix_iPhone[0:3, 0:3] = rotM_iPhone
    matrix_iPhone[0,3] = tvec_iPhone[0]
    matrix_iPhone[1,3] = tvec_iPhone[1] 
    matrix_iPhone[2,3] = tvec_iPhone[2]
    matrix_iPhone[3,3] = 1
    
    matrix_ZED[0:3, 0:3] = rotM_ZED
    matrix_ZED[0,3] = tvec_ZED[0]
    matrix_ZED[1,3] = tvec_ZED[1] 
    matrix_ZED[2,3] = tvec_ZED[2]
    matrix_ZED[3,3] = 1
    
    
    matrix = np.matmul(matrix_iPhone, matrix_ZED)
    
    rotM = matrix[0:3, 0:3]
    rvec2 = cv2.Rodrigues(rotM)[0]
    rvec = np.subtract(rvec_iPhone, rvec_ZED)
    
    
    tvec = np.subtract(tvec_iPhone, tvec_ZED)
    tvec2 = matrix[0:3,3]
    return (rvec, tvec)
    
    

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

#(mtxiPhone, distiPhone, _, _) = getCameraIntrinsic(images_iPhone)
#rvecs, tvecs = getCameraExtrinsic(imagesARKit, mtxiPhone, distiPhone)
# used to get extrinsic ZED
#images_ZED = ["./UNO/samples/RGB_0.png",
#             "./DUE/samples/RGB_0.png",
#             "./TRE/samples/RGB_0.png",
#             "./QUATTRO/samples/RGB_0.png"]

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


images_ZED = ["./test1/ZED/RGB_0.png"]
images_ZED = ["./test_iPhone_offset_x_45_degrees/IMG_2023.JPG"]
images_iPhone = ["./test1/iOS/IMG_2012.jpg"]
images_iPhone =  glob.glob('C:\\Users\\Fulvio Bertolini\\VGIS8\\cameraCalibration\\basicRotations\\IMG_*.jpg')

images_iPhone =  [ "./basicRotations/IMG_2014.jpg", "./basicRotations/IMG_2015.jpg", "./basicRotations/IMG_2016.jpg", "./basicRotations/IMG_2017.jpg"]
images_iPhone = ["./test_iPhone_offset_x_45_degrees/IMG_2024.JPG"]

rvecs_ZED, tvecs_ZED = getCameraExtrinsic(images_ZED, cameraMatrix_iPhone, distCoeff_iPhone)
rvecs_iPhone, tvecs_iPhone = getCameraExtrinsic(images_iPhone, cameraMatrix_iPhone, distCoeff_iPhone)

#
#rotMat_iPhone_2_CheckerBoard = cv2.Rodrigues(rvecs_iPhone[0])[0]
#rotMat_ZED_2_CheckerBoard = cv2.Rodrigues(rvecs_ZED[0])[0]
#rotMat_CheckerBorard_2_ZED = np.transpose(rotMat_ZED_2_CheckerBoard)
#
#rotMat_iPhone_2_ZED = np.matmul(rotMat_iPhone_2_CheckerBoard, rotMat_CheckerBorard_2_ZED)
#
#
#transVec_iPhone_2_CheckerBoard = tvecs_iPhone[0]
#transVec_ZED_2_CheckerBoard = tvecs_ZED[0]
#transVec_CheckerBorard_2_ZED = np.multiply(transVec_ZED_2_CheckerBoard, -1)
#
#transVec_iPhone_2_ZED = np.add(transVec_iPhone_2_CheckerBoard, transVec_CheckerBorard_2_ZED)
#
#
#
#
#
#transfMat_iPhone_2_ZED = np.zeros((4,4), np.float)
#
#transfMat_iPhone_2_ZED[0:3, 0:3] = rotMat_iPhone_2_ZED
#transfMat_iPhone_2_ZED[0,3] = transVec_iPhone_2_ZED[0]
#transfMat_iPhone_2_ZED[1,3] = transVec_iPhone_2_ZED[1] 
#transfMat_iPhone_2_ZED[2,3] = transVec_iPhone_2_ZED[2]
#transfMat_iPhone_2_ZED[3,3] = 1
#
#rotMat_iPhone_2_ZED_transposed = np.transpose(rotMat_iPhone_2_ZED)
#  
#eulerAngles_transposed = cv2.Rodrigues(rotMat_iPhone_2_ZED_transposed)[0]  
#eulerAngles = cv2.Rodrigues(rotMat_iPhone_2_ZED)[0]

eulerAngles_ARKit_2_ZED, tranlation_ARKit_2_ZED = computeARKit_2_ZED_matrix(rvecs_iPhone[0], tvecs_iPhone[0], rvecs_ZED[0], tvecs_ZED[0])
print ("euler angles:\n", np.multiply(180/np.pi,eulerAngles_ARKit_2_ZED))
print("translation:\n", tranlation_ARKit_2_ZED)

print("ah scemooooo")




