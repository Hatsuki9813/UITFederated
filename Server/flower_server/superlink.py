from flask import Flask
import mysql.connector
import subprocess
import json


def startsuperlink():
    command = [
        "flower-superlink",
        "--ssl-ca-certfile", "certificates/ca.crt",
        "--ssl-certfile", "certificates/server.pem",
        "--ssl-keyfile", "certificates/server.key"
    ]
    subprocess.Popen(command)
startsuperlink()