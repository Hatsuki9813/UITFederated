from flask import Flask
import mysql.connector
import subprocess
import json


def startsuperlink():
    command = ["flower-superlink", "--insecure"]
    subprocess.Popen(command)
startsuperlink()