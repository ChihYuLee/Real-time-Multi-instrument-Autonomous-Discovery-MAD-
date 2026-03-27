import socket
import threading
import subprocess
from pathlib import Path
import os 
import base64

from helper_07132025.config import *
from helper_07132025.utils import *
from helper_07132025.gadds import GADDS, GADDS_SLM

class Server:
    def __init__(self, address, port=None) -> None:
        self.server = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.server.settimeout(None)
        self.address = address if address is not None else self.get_local_address()
        self.port = port
        try:
            self.server.bind( (self.address, self.port) )
        except socket.error as e:
            self.server.close()
            raise e

        self.threads = []
    
    def handle(self, connection, address):
        connected = True
        while connected:
            try:
                msg = get_message(connection)
                if msg is None:
                    connected = False
                elif msg['msg'] == DISCONNECT_MSG: 
                    connected = False
                elif msg['msg'] == GADDS_MSG:
                    gadds = GADDS(**json.loads(msg['content']))

                    # generate commands 
                    cmd = gadds.get_command()
                    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) as p:
                        p_return_code = p.wait()
                        p_stdout, p_stderr = p.communicate()
                        if isinstance(p_stdout, bytes):
                            p_stdout = p_stdout.decode(ENCODE_FORMAT)
                        if isinstance(p_stderr, bytes):
                            p_stderr = p_stderr.decode(ENCODE_FORMAT)

                    # read file and return
                    xrd_measurement_plt = gadds.get_return_file()
                    if Path(xrd_measurement_plt).exists():
                        with open(xrd_measurement_plt, "r") as f:
                            plt_text = f.read()
                    else:
                        plt_text = ""
                    response = {
                        "measurement": plt_text,
                        "GADDS": {
                            "return_code":p_return_code, 
                            "stdout" : p_stdout,
                            "stderr" : p_stderr,
                        }
                    }
                    send_message(connection, msg="GADDS", content=response)

                elif msg['msg'] == SLM_MSG:
                    # receive slm file 
                    print('----SLM file is coming----')
                    try:
                        content = json.loads(msg['content'])  # Convert JSON string to dictionary
                        filename = content.get("filename")  # Extract filename--> string 
                        file_data = content.get("data") #Extract content ---> string 
                        #print( filename+ ' is sent from client')
                    except json.JSONDecodeError:
                        print("Error: Received content is not valid JSON.")
                    # passing sample number and file 
                    gadds= GADDS_SLM(filename, file_data)
                    cmd = gadds.get_command()
                                
                    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) as p:
                        p_return_code = p.wait()
                        p_stdout, p_stderr = p.communicate()
                        if isinstance(p_stdout, bytes):
                            p_stdout = p_stdout.decode(ENCODE_FORMAT)
                        if isinstance(p_stderr, bytes):
                            p_stderr = p_stderr.decode(ENCODE_FORMAT)
                    

                    print('--- All 3 frames are being generated ---')
                    #generate 3 .gfrms and send seperately, example name: FGT_001_001
                    xrd_measurement_plt = gadds.get_return_file()
                    for i in range(FRAME): 
                        if Path(xrd_measurement_plt[i]).exists():
                            print(xrd_measurement_plt[i])
                            file_size = os.path.getsize(xrd_measurement_plt[i])
                            with open(xrd_measurement_plt[i], "rb") as f:
                                plt_binary = f.read(file_size) #binary (byte)
                                encoded_data = base64.b64encode(plt_binary).decode(ENCODE_FORMAT) #string 
                                plt_text= encoded_data
                                print('---  '+Path(xrd_measurement_plt[i]).name+' is sent ---')       
                        else:
                            plt_text = ""
                    # delete thee strings 
                        p_return_code=''
                        p_stdout= ''
                        p_stderr= ''
                        response = {
                            "filename": Path(xrd_measurement_plt[i]).name,
                            "measurement": plt_text,
                            "GADDS": {
                                "return_code":p_return_code, 
                                "stdout" : p_stdout,
                                "stderr" : p_stderr,
                            }
                        }
                        send_message(connection, msg="GFRAM", content=response)
                        
                    
                elif msg['msg'] == COMMAND_MSG and DEBUG:
                    cmd = msg['content']['command']
                    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) as p:
                        p_return_code = p.wait()
                        p_stdout, p_stderr = p.communicate()
                        if isinstance(p_stdout, bytes):
                            p_stdout = p_stdout.decode(ENCODE_FORMAT)
                        if isinstance(p_stderr, bytes):
                            p_stderr = p_stderr.decode(ENCODE_FORMAT)


                    response = {
                        "return_code" : p_return_code,
                        "stdout" : p_stdout,
                        "stderr" : p_stderr,
                    }
                    send_message(connection, msg="Command", content=response)
                else:
                    pass
            except Exception as e:
                send_message(connection, msg="exception", content={'error_msg':str(e)})
                    
        connection.close()
        print("[DISCONNECT] Client {}:{}".format(address[0], address[1]))
        
    def host(self):
        self.server.listen(BACKLOG)
        print("[START] Listening on {}:{}".format(self.address, self.port))
        while True:
            conn, addr = self.server.accept()
            thread = threading.Thread(target=self.handle, args=(conn, addr))
            thread.start()
            print("[CONNECT] Client {}:{}".format(addr[0], addr[1]))
            print("[ACTIVE CONNECTIONS] {}".format(threading.active_count() - 1))
                
                
    def get_local_address(self):
        return socket.gethostbyname(socket.gethostname())

if __name__ == "__main__":
    server = Server(port=PORT_XRD)
    try:
        server.host()
    except KeyboardInterrupt:
        print("[CLOSE] detect Keyboard Inter, close the server")
        server.server.close()