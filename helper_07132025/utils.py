import json
from helper_07132025.config import *
import re 
import os 
import time 

def create_header(message, header_size=HEADER_SIZE):
    message_length = len(message)
    header = str(message_length).zfill(header_size)  # Fills with zeros instead of spaces
    header = header.encode(ENCODE_FORMAT)
    
    assert len(header) == header_size  # Ensures correct size
    return header

def parser_header(conn, header_size=HEADER_SIZE):
    header = conn.recv(header_size).decode(ENCODE_FORMAT)
    if header:
        message_length = int(header)
        return message_length
    else:
        print('Not the header size.')
        return None


def get_message_gfrm(conn, header_size=HEADER_SIZE):
    message_length = parser_header(conn, header_size=header_size)
    if not message_length:
        return {"status": 0, "msg": "exception", "content": {"error_msg": "Invalid header size"}}
    try:
        data = b""
        while len(data) < message_length:
            chunk = conn.recv(min(4096, message_length - len(data)))  # Read in 4KB chunks
            if not chunk:
                break
            data += chunk
        
        raw_message = data.decode(ENCODE_FORMAT)  # Decode JSON

        message = json.loads(raw_message)  # Parse JSON
        return message
    except json.JSONDecodeError:
        return {"status": 0, "msg": "exception", "content": {"error_msg": "Invalid JSON format"}}

    except Exception as e:
        return {"status": 0, "msg": "exception", "content": {"error_msg": str(e)}}



def get_message(conn, header_size=HEADER_SIZE):
    message_length = parser_header(conn, header_size=header_size)
    if message_length:
        message = conn.recv(message_length).decode(ENCODE_FORMAT) # received json
        try:
            message = json.loads(message)
            return message
        except json.JSONDecodeError as e:
            return {"status": 0, "msg": "exception", "content": {"error_msg": "Invalid JSON format"}}
    else:
        print("Failed to get message length.")
        return None
        
        
def send_message(conn, msg="", content={}, status=0, header_size=HEADER_SIZE):
    message = {
        "status" : status,
        "msg" : msg,
        "content" : content
    }
    message = json.dumps(message).encode(ENCODE_FORMAT)    
    header = create_header(message, header_size=header_size)
    
    conn.send(header)
    conn.send(message)

def create_slm(x: int,y: int,z: int, sample_num: int, slm_template= SLM_REF)-> str:
    """
    To help create a slm file for sending 
    return a file path 
    """

    with open(slm_template, 'r') as f:
        template_content = f.read()
    template_content = template_content.replace("%1", str(x))
    template_content = template_content.replace("%2", str(y))
    template_content = template_content.replace("%3", str(z))
    template_content = template_content.replace("%4", str(sample_num))
    template_content = re.sub(r"%\d", "", template_content)  
    template_content = re.sub(r"^!.*\n", "", template_content, flags=re.MULTILINE)
    template_content = template_content.lstrip('\n')
    output_file_path = XRD_FOLDER+ str(sample_num)+'.slm'
    with open(output_file_path, 'w') as f:
        f.write(template_content)
    return output_file_path 

def generate_mst_list(sample_number):
    return [XRD_FOLDER+f"MST_{sample_number:03d}_{i:03d}.gfrm" for i in range(1, 4)]



    
    