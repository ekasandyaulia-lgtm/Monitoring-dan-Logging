from prometheus_client import start_http_server, Gauge, Counter
import random, time

# --- Gauges (value changing) ---
accuracy = Gauge('model_accuracy', 'Model Accuracy')
latency = Gauge('model_latency', 'Inference Latency (ms)')
cpu = Gauge('cpu_usage', 'CPU Usage (%)')
memory = Gauge('memory_usage', 'Memory Usage (%)')
throughput = Gauge('throughput', 'Throughput (req/s)')
error_rate = Gauge('error_rate', 'Error Rate (%)')

# --- Counters ---
requests_total = Counter('request_count', 'Number of Requests')
successful_requests = Counter('successful_requests', 'Number of Successful Requests')
failed_requests = Counter('failed_requests', 'Number of Failed Requests')
retries = Counter('retries', 'Number of Retries')
alerts_triggered = Counter('alerts_triggered', 'Number of Alerts Triggered')

start_http_server(8000)  # Prometheus scrape port

while True:
    accuracy.set(random.uniform(0.7, 0.95))
    latency.set(random.uniform(50, 250))
    cpu.set(random.uniform(10, 80))
    memory.set(random.uniform(30, 90))
    throughput.set(random.uniform(50, 500))
    error_rate.set(random.uniform(0, 5))

    requests_total.inc(random.randint(1,5))
    successful_requests.inc(random.randint(1,5))
    failed_requests.inc(random.randint(0,2))
    retries.inc(random.randint(0,2))
    alerts_triggered.inc(random.randint(0,1))

    time.sleep(5)