# server.py

from xmlrpc.server import SimpleXMLRPCServer

def concatenate(str1, str2):
    return str1 + str2

server = SimpleXMLRPCServer(("localhost", 8000))
print("Server running...")

server.register_function(concatenate, "concatenate")

server.serve_forever()