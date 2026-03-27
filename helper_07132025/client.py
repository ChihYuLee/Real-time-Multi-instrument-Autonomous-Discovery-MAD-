import socket
from helper_07132025.config import *
from helper_07132025.utils import get_message, send_message, get_message_gfrm, create_slm
import json
import pandas as pd
import os 
import base64

class Client:

    def __init__(self, host_address, port) -> None:
        self.client = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.client.settimeout(None)
        self.host_address = host_address
        self.port = int(port) 
        self.client.connect( (self.host_address, self.port) )


    def send(self, msg, content):
        send_message(self.client, msg, content)
        #response = get_message(self.client)        
        #return response


    def send_command(self, command):
        msg = COMMAND_MSG
        content = {"command":command}
        return self.send(msg, content)


    def send_gadds(self, gadds):
        msg = GADDS_MSG
        content = json.dumps(gadds.to_dict())
        return self.send(msg, content)

    def receive(self):
        msg= get_message(self.client)
        return msg

    def receive_gfrm(self):
        msg= get_message_gfrm(self.client)
        if "content" in msg:
            file_name= msg["content"]["filename"]
            received_data = msg["content"]["measurement"]
            encoded_data= base64.b64decode(received_data)
            with open(XRD_FOLDER+ file_name, "wb") as file:
                file.write(encoded_data)
                print('---  '+file_name+ '  arriving---')
        else:
            print('--- No file coming--- ')


    def receive_csv(self):
        time.sleep(PROBE_TIME+ BUFFER)
        msg= get_message_gfrm(self.client)
        if "content" in msg:
            file_name= msg["content"]["filename"]
            received_data = msg["content"]["measurement"]
            with open(PROBE_FOLDER+ file_name, "w") as file:
                file.write(received_data)
                print('---  '+file_name+ '  arriving---')
        else:
            print('--- No file coming--- ')
        

    def close(self):
        send_message(self.client, msg=DISCONNECT_MSG)

    def send_measure_xrd(self, num: int):
        """
        wrap up function for xrd command sending
        1. generate .slm file 
        2. send out the .slm file 
        """
        df = pd.read_csv(XRD_TARGET, sep='\s+', header=None)
        array = df.to_numpy()
        x= array[num,2]
        y= array[num,3]
        z= array[num,4]
        if z> XRD_Z_LIMIT:
            print('Dangerous Z')
        else:
            slm_file= create_slm(x, y, z, num)
            self.send_slm(slm_file)
            print('File Sent')

    def send_slm(self, slm:str):
        """
        send slm file to server
        """
        msg = SLM_MSG
        with open(slm, "r") as file:
            file_content = file.read()
        # dictionary 
        final = {"filename": os.path.basename(slm), "data": file_content}
        # string 
        content = json.dumps(final)
        return self.send(msg, content)

    
    def send_measure_probe(self, num:int):
        """
        wrap up function for probe measurement command 
        """
        df = pd.read_csv(PROBE_TARGET, sep='\s+', header=None)
        array = df.to_numpy()
        x= array[num,1]
        y= array[num,2]

        msg= PROBE_MSG
        final= {'x': x, 'y': y, 'id': num}
        content = json.dumps(final)
        return self.send(msg, content)


if __name__ == "__main__":
    client = Client()
    try:
        while True:
            command = input("cmd: ")
            respond = client.send_command(command)
            print("Returncode: {} \n".format(respond['content']['return_code']))
            print(respond['content']['stdout'])
            print(respond['content']['stderr'])
    except KeyboardInterrupt as e:
        client.close()
        print()
        print("Detect KeyboardInterrupt. Close the client.")

