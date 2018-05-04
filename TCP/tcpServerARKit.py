# -*- coding: utf-8 -*-
"""
Created on Thu May  3 19:15:25 2018

@author: Fulvio Bertolini
"""

import socket
import numpy as np

def sendTo(ip, port, data):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))
    s.send(data)
    
def decodeMatrix(string):

    matrix = np.zeros((4,4), np.float)
    
    rows = string.split('\n')
    
    cols_row0 = rows[0].split(' ')
    cols_row1 = rows[1].split(' ')
    cols_row2 = rows[2].split(' ')
    cols_row3 = rows[3].split(' ')
    
    
    for i in range(0,4):
        for j in range(0,4):
            matrix[i,j] = (((string.split('\n'))[i]).split())[j]
            
    m00 = cols_row0[0]
    m01 = cols_row0[1]
    m02 = cols_row0[2]
    m03 = cols_row0[3]
    m10 = cols_row1[0]
    m11 = cols_row1[1]
    m12 = cols_row1[2]
    m13 = cols_row1[3]
    m20 = cols_row2[0]
    m21 = cols_row2[1]
    m22 = cols_row2[2]
    m23 = cols_row2[3]
    m30 = cols_row3[0]
    m31 = cols_row3[1]
    m32 = cols_row3[2]
    m33 = cols_row3[3]
    
    print("m00: ", m00)
    print("m01: ", m01)
    print("m02: ", m02)
    print("m03: ", m03)
    print("m10: ", m10)
    print("m11: ", m11)
    print("m12: ", m12)
    print("m13: ", m13)
    print("m20: ", m20)
    print("m21: ", m21)
    print("m22: ", m22)
    print("m23: ", m23)
    print("m30: ", m30)
    print("m31: ", m31)
    print("m32: ", m32)
    print("m33: ", m33)
    
def Main(ip, port):
    
    s = socket.socket()
    s.bind((ip,port))
    s.listen(1)
    print("listening")
    c, addr = s.accept()
    print ("Connection from: " + str(addr))
    while 1:
        receivedData = c.recv(1024)
        if not receivedData:
            break
        receivedDataStr = receivedData.decode('utf-8')
        decodeMatrix(receivedDataStr)
        if(receivedDataStr == "stop"):
            print("stopping...")
            break
        print("from connected user:\n" + receivedDataStr)
        sendData = "TUA mamma".encode('utf-8')
        c.send(sendData)
        #c.close()
    s.shutdown(socket.SHUT_RDWR)
    s.close()
    print("shut down")

if __name__ == '__main__':
   
    ipHost = "172.20.10.2"
    portHost = 5002
    Main(ipHost, portHost)
