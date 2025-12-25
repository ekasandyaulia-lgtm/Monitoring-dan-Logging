from prometheus_client import start_http_server
import time

# Metrics are exposed from inference.py
# This file exists only for documentation purpose

if __name__ == "__main__":
    start_http_server(8001)
    while True:
        time.sleep(10)