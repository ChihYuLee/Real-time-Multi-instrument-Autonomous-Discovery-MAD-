from helper_07132025.config import *
import socket 
import threading 
import numpy as np
import pickle


class SocketServer:
    def __init__(self, host, port, buffer_size=1024):
        self.host = SERVER
        self.port = PORT
        self.buffer_size = buffer_size
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)  # The server can listen for 5 simultaneous connections
        print(f"Server listening on {self.host}:{self.port}")

    def handle_client(self, client_socket, client_address):
        """Handle client connection in a separate thread."""
        print(f"Connection established with {client_address}")

        # Example of sending a message to the client
        client_socket.sendall("Welcome to the server!".encode())

        # Receive the file from the client
        self.receive_file(client_socket, 'received_file_from_' + str(client_address[1]) + '.csv')

        # Example of sending a message to the client after receiving the file
        client_socket.sendall("File received successfully!".encode())

        # Close the client connection
        client_socket.close()
        print(f"Connection closed with {client_address}")

    def accept_connections(self):
        """Accept and handle multiple client connections concurrently."""
        while True:
            # Accept a connection from a client
            client_socket, client_address = self.server_socket.accept()
            # Start a new thread to handle the client
            client_thread = threading.Thread(target=self.handle_client, args=(client_socket, client_address))
            client_thread.start()

    def receive_file(self, client_socket, output_filename):
        """Receive file from the client and save it."""
        with open(output_filename, 'wb') as file:
            print(f"Receiving file from {client_socket.getpeername()}")
            while True:
                data = client_socket.recv(self.buffer_size)
                if not data:
                    break
                file.write(data)
        print(f"File received successfully from {client_socket.getpeername()}!")

    def send_message(self, client_socket, message):
        """Send a message to a specific client."""
        client_socket.sendall(message.encode())
        print(f"Message sent to {client_socket.getpeername()}: {message}")

    def close_server(self):
        """Close the server socket."""
        self.server_socket.close()
        print("Server socket closed.")



class SocketClient:
    def __init__(self, buffer_size=4096):
        self.host = SERVER
        self.port = PORT
        self.buffer_size = buffer_size
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect_to_server(self):
        # Connect to the server
        self.client_socket.connect((self.host, self.port))
        print(f"Connected to server at {self.host}:{self.port}")

    def receive_message(self):
        """Receive and decode a message (it could be a NumPy array or a regular message)."""
        try:
            # Receive the data from the server
            data = self.client_socket.recv(1024)  # Adjust buffer size as needed
            if data:
                # Try to deserialize the data
                try:
                    # Deserialize the data as a potential NumPy array
                    message = pickle.loads(data)
                    if isinstance(message, np.ndarray):  # Check if it's a NumPy array
                        print(f"Received NumPy array: {message}")
                    else:
                        print(f"Received object: {message}")
                except pickle.UnpicklingError:
                    # If it's not a NumPy array, it might be a regular message
                    try:
                        message = data.decode('utf-8')  # Try decoding as a string
                        print(f"Received regular message: {message}")
                    except UnicodeDecodeError:
                        print("Received non-UTF-8 binary data.")
        except Exception as e:
            print(f"Error receiving message: {e}")

    def send_message(self, message):
        """Send a regular message or NumPy array to the server."""
        try:
            # If the message is a NumPy array, we serialize it
            if isinstance(message, np.ndarray):
                serialized_message = pickle.dumps(message)
                self.client_socket.send(serialized_message)
                print(f"Sent NumPy array: {message}")
            else:
                # Otherwise, send it as a regular message (string)
                self.client_socket.send(message.encode('utf-8'))
                print(f"Sent regular message: {message}")
        except Exception as e:
            print(f"Error sending message: {e}")
    
    def send_file(self, filename):
        """Send a file to the server."""
        try:
            # Send the initial message indicating a file transfer
            self.client_socket.send(f"FILE:{filename}".encode('utf-8'))  # Indicate to server that a file is coming
            
            # Open the file to be sent
            with open(filename, 'rb') as file:
                while chunk := file.read(1024):  # Read the file in 1 KB chunks
                    self.client_socket.send(chunk)  # Send each chunk to the server

            # Optionally, wait for server confirmation that the file was received
            confirmation = self.client_socket.recv(1024)  # Receive server's confirmation message
            print(f"Server response: {confirmation.decode('utf-8')}")
        
        except Exception as e:
            print(f"Error sending file: {e}")
        #finally:
            # Close the connection once the file transfer is complete
        #    self.client_socket.close()

    def close_connection(self):
        """Close the connection to the server."""
        self.client_socket.close()
        print("Connection closed.")




