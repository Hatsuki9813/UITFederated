import subprocess

def startsuperlink():
    command = ["flower-superlink", "--insecure"]
    process = subprocess.Popen(command)

if __name__ == '__main__':
    startsuperlink()
