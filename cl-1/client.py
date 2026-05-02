# client.py

import xmlrpc.client

try:
    # Connect to server
    proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")

    num = int(input("Enter a number: "))
    response = proxy.factorial(num)

    if response["status"] == "error":
        print("Error:", response["message"])

    else:
        if response["type"] == "full":
            print("\nFactorial is:")
            print(response["result"])

        elif response["type"] == "approx":
            print("\n" + response["message"])
            print("Number of digits:", response["digits"])

except Exception as e:
    print("Client Error:", e)