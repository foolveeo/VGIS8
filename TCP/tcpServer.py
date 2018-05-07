import socket

def sendTo(ip, port, data):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))
    s.send(data)
   
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
        
        
        
        fileTxt = open("./ARKitWorld_2_ARKitCam.txt", "a+")
        fileTxt.write(receivedDataStr)
        fileTxt.close()
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

