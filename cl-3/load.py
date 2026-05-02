import random
import time

# Simulated servers
servers = {
    "Server1": 0,
    "Server2": 0,
    "Server3": 0
}

# -------------------------------
# ROUND ROBIN
# -------------------------------
def round_robin(requests):
    print("\n--- Round Robin Load Balancing ---")
    server_list = list(servers.keys())
    
    for i, req in enumerate(requests):
        server = server_list[i % len(server_list)]
        print(f"Request {req} → {server}")
        time.sleep(0.3)


# -------------------------------
# LEAST CONNECTIONS
# -------------------------------
def least_connections(requests):
    print("\n--- Least Connections Load Balancing ---")
    
    # Track active connections
    server_load = {
        "Server1": [],
        "Server2": [],
        "Server3": []
    }

    for req in requests:
        # Remove completed requests
        for server in server_load:
            server_load[server] = [r for r in server_load[server] if r > 0]

        # Choose server with least active requests
        server = min(server_load, key=lambda s: len(server_load[s]))

        print(f"Request {req} → {server}")

        # Assign random duration (simulate processing time)
        duration = random.randint(1, 3)
        server_load[server].append(duration)

        # Decrease duration of all active requests
        for s in server_load:
            server_load[s] = [r-1 for r in server_load[s]]


# -------------------------------
# SIMULATE CLIENT REQUESTS
# -------------------------------
requests = [f"Req{i}" for i in range(1, 11)]

print("Incoming Requests:", requests)

round_robin(requests)
least_connections(requests)