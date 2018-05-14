import cv2
import numpy as np
from matplotlib import pyplot as plt


def bgr_to_chR_chG(bgrImg):
    rows, cols, _ = bgrImg.shape
    
    chR = np.zeros((rows, cols), np.float)
    chG = np.zeros((rows, cols), np.uint8)
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


img = cv2.imread('RGB_29.png')

chR, chB, chG = bgr_to_chR_chG(img)


edges = cv2.Canny(chG,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()