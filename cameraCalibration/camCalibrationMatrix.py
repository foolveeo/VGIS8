# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:46:37 2018

@author: Fulvio Bertolini
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
  
def getCameraMatrix(rvec, tvec):
    Cam_2_Chkb = np.zeros((4,4), np.float)
    Chkb_2_Cam = np.zeros((4,4), np.float)
    
    
    rotM_Cam_2_Chkb = cv2.Rodrigues(rvec)[0]
    
    Cam_2_Chkb[0:3, 0:3] = rotM_Cam_2_Chkb
    Cam_2_Chkb[0:3, 3] = tvec.ravel()
    Cam_2_Chkb[3,3] = 1
    
    Chkb_2_Cam = np.linalg.inv(Cam_2_Chkb)
    
    
    return Cam_2_Chkb, Chkb_2_Cam
    
    
    
    
    
    
def writeMatrix(matrix):
    
    string:str = ""
    for i in range(0,matrix.shape[0]):
        for j in range(0,matrix.shape[1]):
            string += "{:.9f}".format(matrix[i,j])
            if(j != 3):
                string += " "
        if(i != 3):
                string += "\t"
    
    return string

def multiplyMatrices4x4(rvec_iPhone, tvec_iPhone, rvec_ZED, tvec_ZED):
    
    print("iPhone:\nrvec:\n ", rvec_iPhone, "\ntvec: \n", tvec_iPhone)
    print("\n")
    print("ZED:\nrvec:\n ", rvec_ZED, "\ntvec: \n", tvec_ZED)
    print("\n")
    
    rvec_iPhone[1,0] = rvec_iPhone[1,0] * -1
    rvec_ZED[1,0] = rvec_ZED[1,0] * -1
    tvec_iPhone[1] = tvec_iPhone[1] * -1
    tvec_ZED[1] = tvec_ZED[1] * -1
    rotM_iPhone = cv2.Rodrigues(rvec_iPhone)[0]
    rotM_ZED = cv2.Rodrigues(rvec_ZED)[0]
    
    rotM_iPhone_inverse = np.linalg.inv(rotM_iPhone)
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
    ChB_2_iPhone = np.linalg.inv(iPhone_2_ChB)
    iPhone_2_ZED = np.matmul(iPhone_2_ChB, ChB_2_ZED)
    
    ZED_2_iPhone = np.matmul(ZED_2_ChB, ChB_2_iPhone)
    
    
    tvec = iPhone_2_ZED[0:3,3]
    rvec = cv2.Rodrigues(iPhone_2_ZED[0:3,0:3])[0]
    
    tvec_inv = ZED_2_iPhone[0:3,3]
    rvec_inv = cv2.Rodrigues(ZED_2_iPhone[0:3,0:3])[0]

    origin = np.zeros((4,1), np.float)
   # origin[0:3] = tvec_iPhone
    #origin[-1] = origin[1] * -1 
    origin[3] = 1
    print("checkerboard origin in checkerboard coord (according to iPhone_2Chk): ", np.matmul(ChB_2_iPhone, origin))
    iPhone2ZED:np.ndarray((4,4), np.float32) = np.matmul(ZED_2_ChB, ChB_2_iPhone)
    iPhone2ZED_inv = np.linalg.inv(iPhone2ZED)
    print("\nzed transf\ntvec: ", np.matmul(iPhone2ZED, origin))
    print("rvec:\n ", cv2.Rodrigues(iPhone2ZED[0:3,0:3])[0])
    return rvec, tvec, iPhone2ZED_inv
  
def computeARKit_2_ZED_matrix(rvec_iPhone, tvec_iPhone, rvec_ZED, tvec_ZED):

    rvec, tvec, matrix = multiplyMatrices4x4(rvec_iPhone, tvec_iPhone, rvec_ZED, tvec_ZED)
    
    
    return rvec, tvec, matrix
    
    

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
            cv2.resizeWindow("img", 2400, 1200)              # Resize window to specified dimensions
    
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners,ret)
            cv2.imshow('img',img)
            cv2.waitKey(10)
    
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


images_ZED = [ "./calibrationDebugImages/ZED/A/samples/RGB_0.PNG",
                  "./calibrationDebugImages/ZED/B/samples/RGB_0.PNG",
                  "./calibrationDebugImages/ZED/C_30/samples/RGB_0.PNG",
                  "./calibrationDebugImages/ZED/C_45/samples/RGB_0.PNG",
                  "./calibrationDebugImages/ZED/D_-30/samples/RGB_0.PNG",
                  "./calibrationDebugImages/ZED/D_-45/samples/RGB_0.PNG",
                  "./calibrationDebugImages/ZED/E/samples/RGB_0.PNG",
                  "./calibrationDebugImages/ZED/F_15/samples/RGB_0.PNG",
                  "./calibrationDebugImages/ZED/F_30/samples/RGB_0.PNG",
                  "./calibrationDebugImages/ZED/G_-10/samples/RGB_0.PNG",
                  "./calibrationDebugImages/ZED/G_-20/samples/RGB_0.PNG" ]




images_iPhone = ["./calibrationDebugImages/iPhone/A.JPG",
					"./calibrationDebugImages/iPhone/B.JPG",
					"./calibrationDebugImages/iPhone/C_30.JPG",
					"./calibrationDebugImages/iPhone/C_45.JPG",
					"./calibrationDebugImages/iPhone/D_-30.JPG",
					"./calibrationDebugImages/iPhone/D_-45.JPG",
					"./calibrationDebugImages/iPhone/E.JPG",
					"./calibrationDebugImages/iPhone/F_15.JPG",
					"./calibrationDebugImages/iPhone/F_30.JPG",
					"./calibrationDebugImages/iPhone/G_-10.JPG",
					"./calibrationDebugImages/iPhone/G_-20.JPG" ]


rvecs_ZED, tvecs_ZED = getCameraExtrinsic(images_ZED, cameraMatrix_ZED, distCoeff_ZED)
rvecs_iPhone, tvecs_iPhone = getCameraExtrinsic(images_iPhone, cameraMatrix_iPhone, distCoeff_iPhone)

originZED = np.zeros((len(images_ZED),4,1), np.float)
origin_iPhone = np.zeros((len(images_iPhone),4,1), np.float)
originZED[:,3,0] = 1
origin_iPhone[:,3,0] = 1

pointsZED = np.zeros((len(images_ZED),4,1), np.float)
points_iPhone = np.zeros((len(images_iPhone),4,1), np.float)
rotationsCkb_2_ZED = np.zeros((len(images_ZED),3,1), np.float)
rotationsCkb_2_iPhone = np.zeros((len(images_iPhone),3,1), np.float)
rotationsZED_2_Ckb = np.zeros((len(images_ZED),3,1), np.float)
rotations_iPhone_2_Ckb = np.zeros((len(images_iPhone),3,1), np.float)
ZEDCam_2_Chkb = np.zeros((len(images_ZED),4,4), np.float)
iPhoneCam_2_Chkb = np.zeros((len(images_iPhone),4,4), np.float)
Chkb_2_ZEDCam = np.zeros((len(images_ZED),4,4), np.float)
Chkb_2_iPhoneCam = np.zeros((len(images_iPhone),4,4), np.float)


for i in range(0,11):
    ZEDCam_2_Chkb[i,:,:], Chkb_2_ZEDCam[i,:,:] = getCameraMatrix(rvecs_ZED[i], tvecs_ZED[i])
    iPhoneCam_2_Chkb[i,:,:], Chkb_2_iPhoneCam[i,:,:] = getCameraMatrix(rvecs_iPhone[i], tvecs_iPhone[i])
    
    pointsZED[i,:,:] = np.matmul(ZEDCam_2_Chkb[i,:,:], originZED[i,:,:])
    originZED[i,:,:] = np.matmul(Chkb_2_ZEDCam[i,:,:], pointsZED[i,:,:])
    points_iPhone[i,:,:] = np.matmul(iPhoneCam_2_Chkb[i,:,:], origin_iPhone[i,:,:])
    origin_iPhone[i,:,:] = np.matmul(Chkb_2_iPhoneCam[i,:,:], points_iPhone[i,:,:])
    
    rotationsZED_2_Ckb[i,:,:] = cv2.Rodrigues(Chkb_2_ZEDCam[i,0:3, 0:3])[0]
    rotationsCkb_2_ZED[i,:,:] = cv2.Rodrigues(ZEDCam_2_Chkb[i,0:3, 0:3])[0]
    rotations_iPhone_2_Ckb[i,:,:] = cv2.Rodrigues(Chkb_2_iPhoneCam[i,0:3, 0:3])[0]
    rotationsCkb_2_iPhone[i,:,:] = cv2.Rodrigues(iPhoneCam_2_Chkb[i,0:3, 0:3])[0]

origin_iPhone_G_20 = np.zeros((4,1), np.float)
G_20_ZED_D_45 = np.zeros((4,1), np.float)
origin_iPhone_G_20[3,0] = 1
G_20_ZED_D_45[3,0] = 1
G_20_ZED_D_45 = np.matmul(Chkb_2_iPhoneCam[10,:,:], origin_iPhone_G_20)
G_20_ZED_D_45 = np.matmul(ZEDCam_2_Chkb[5,:,:], G_20_ZED_D_45)


xDir_A = np.zeros((4,1), np.float)
xDir_ZED_D_45 = np.zeros((4,1), np.float)
xDir_A[0,0] = 1
xDir_ZED_D_45 = np.matmul(Chkb_2_iPhoneCam[0,:,:], xDir_A)
xDir_ZED_D_45 = np.matmul(ZEDCam_2_Chkb[5,:,:], xDir_ZED_D_45)
print(xDir_ZED_D_45)
## G_-20 in iPhone coordinates to Zed coordinates when zed is in D_-45


#
#eulerAngles_ARKit_2_ZED, translation_ARKit_2_ZED, matrix = computeARKit_2_ZED_matrix(rvecs_iPhone[0], tvecs_iPhone[0], rvecs_ZED[0], tvecs_ZED[0])
#
#print ("euler angles:\n",eulerAngles_ARKit_2_ZED)
#print("translation:\n", translation_ARKit_2_ZED)
#
#matrixFile = open("../TCP/ARKitCam_2_ZEDCam.txt", "a+")
#matrixFile.write(writeMatrix(matrix) + "\n")
#matrixFile.close()
#
print("ah scemooooo")




