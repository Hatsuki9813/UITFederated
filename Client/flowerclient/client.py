import subprocess
import socket
import time
# Lấy địa chỉ IP cục bộ (LAN IP)
server_address = '10.0.145.238'
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Kết nối đến 1 địa chỉ ảo để lấy IP (không thực sự gửi gì)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()

def startsupernode():
    ip_address = get_local_ip()
    command1 = [
    "flower-supernode",
    "--root-certificates", "certificates/ca.crt",
    f"--superlink={server_address}:9092",
    f"--clientappio-api-address={ip_address}:9094",
    "--node-config", "partition-id=0 num-partitions=2"
]
    command2 = [
    "flower-supernode",
    "--root-certificates", "certificates/ca.crt",
    f"--superlink={server_address}:9092",
    f"--clientappio-api-address={ip_address}:9095",
    "--node-config", "partition-id=0 num-partitions=2"
]  
    p1 = subprocess.Popen(command1)
    p2 = subprocess.Popen(command2)
    return p1,p2


def runclient():
    command = ["flwr", "run", ".", "flowerclient"]
    p3 = subprocess.Popen(command)
    return p3

p1,p2 = startsupernode()
time.sleep(10)
p3 = runclient()
'''p3.wait()
p1.terminate()
p2.terminate()'''
