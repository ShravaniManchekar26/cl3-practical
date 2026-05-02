# server.py

from xmlrpc.server import SimpleXMLRPCServer
import math
import sys

sys.set_int_max_str_digits(1000000)

# Safe computation limit
MAX_LIMIT = 1000  

def factorial(n):
    try:
        n = int(n)

        if n < 0:
            return {"status": "error", "message": "Negative numbers not allowed"}

        # ✔ For very large numbers → approximation
        if n > MAX_LIMIT:
            digits = int(n * math.log10(n) - n + 1)
            return {
                "status": "success",
                "type": "approx",
                "message": f"{n}! is too large to compute directly",
                "digits": digits
            }

        # ✔ Normal factorial calculation
        result = 1
        for i in range(2, n + 1):
            result *= i

        return {
            "status": "success",
            "type": "full",
            "result": str(result)  # send as string (RPC safe)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# Start RPC Server
server = SimpleXMLRPCServer(("localhost", 8000))
print("RPC Server running on port 8000...")

server.register_function(factorial, "factorial")

server.serve_forever()