# -*- coding: utf-8 -*-
"""
Created on Thu May  3 19:15:41 2018

@author: Fulvio Bertolini
"""

import socket

   
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
        if(receivedDataStr == "stop"):
            print("stopping...")
            break
        
        print("from connected user:\n" + receivedDataStr)
        
        fileMatix =  open("./matrix.txt", "r")
        linesMatrix = fileMatix.readlines()
        fileMatix.close()
        fileEulerAngles =  open("./eulerAngles.txt", "r")
        linesEuleAngles = fileEulerAngles.readlines()
        fileEulerAngles.close()
        
        sendDataMatrix = linesMatrix[-1]
        sendDataEuler = linesEuleAngles[-1]
        sendDataEncoded = (sendDataEuler + "\n" + sendDataMatrix).encode('utf-8')
        c.send(sendDataEncoded)
        #c.close()
    s.shutdown(socket.SHUT_RDWR)
    s.close()
    print("shut down")

if __name__ == '__main__':
   
    ipHost = "127.0.0.1"
    portHost = 5001
    Main(ipHost, portHost)

