import socket

def Main(ip, port, data):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))
    s.send(data)
   
if __name__ == '__main__':
    
    ipHost = "127.0.0.1"
    portHost = 5002
    string = "La bamba!"
    datToSend = string.encode('utf-8')
    Main(ipHost, portHost, datToSend)

