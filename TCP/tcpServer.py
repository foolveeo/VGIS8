import socket

def sendTo(ip, port, data):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))
    s.send(data)
   
def Main(ip, port):
    
    s = socket.socket()
    s.bind((ip,port))
    s.listen(1)
    c, addr = s.accept()
    print ("Connection from: " + str(addr))
    while 1:
        data = c.recv(1024)
        if not data:
            break
        dataStr = data.decode('utf-8')
        print("from connected user:\n" + dataStr)
        
        #sendTo("127.0.0.1", 5001, data)
    c.close()

if __name__ == '__main__':
   
    ipHost = "192.168.43.161"
    portHost = 5000
    Main(ipHost, portHost)

