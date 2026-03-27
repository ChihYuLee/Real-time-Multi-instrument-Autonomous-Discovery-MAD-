import socket
import threading
import subprocess
from pathlib import Path
import os 
import base64

from helper_07132025.config import *
from helper_07132025.utils import *
from helper_07132025.signatone import *
import pyvisa

class Server:
    def __init__(self, address, port=None) -> None:
        self.server = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.server.settimeout(None)
        self.address = address if address is not None else self.get_local_address()
        self.port = port
        self.stage= pyvisa.ResourceManager().open_resource(STAGE) #stage object, can't be replicated 
        try: 
            response = self.stage.query("*IDN?")
            print("Connected stage successfully! Instrument response:", response)
        except pyvisa.VisaIOError as e:
            print(f"Error: Unable to connect to the instrument. {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        try: 
            probe= pyvisa.ResourceManager().open_resource(PROBE) #probe object, can be replicated  
            response = probe.query("*IDN?")
            print("Connected probe successfully! Instrument response:", response)
        except pyvisa.VisaIOError as e:
            print(f"Error: Unable to connect to the instrument. {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
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
                elif msg['msg'] == PROBE_MSG:
                    content= json.loads(msg['content'])
                    signatone= Signatone(content['x'], content['y'], content['id'], self.stage)
                    success= signatone.execute()
                    if success:
                        print("Execution completed successfully.")
                        csv= signatone.get_return_file()
                        if Path(csv).exists():
                            with open(csv, "r") as f:
                                plt_text = f.read()
                        else:
                            plt_text=''
                    else:
                        print('Execution failed.')
                        plt_text=''

                    response = {"measurement": plt_text, "filename": os.path.basename(csv)}
                    send_message(connection, msg="PROBE", content=response)
                    print('---  '+ os.path.basename(csv)+' is sent ---')   

                    
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

        self.stage.close()   
        print("Instrument connection closed.")        
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
    server = Server()
    try:
        server.host()
    except KeyboardInterrupt:
        print("[CLOSE] detect Keyboard Inter, close the server")
        server.server.close()