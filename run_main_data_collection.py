import os.path
import os
import runpy
import socket
import subprocess
import sys
import time

os.environ['TOWN_NAME'] = "Town12"

try:
    carla_path = "//workspace/tg22/carla_9_15/"
    code_path = ""
    assert os.path.exists(carla_path)
    hpc=False
except:
    carla_path = "//cta/users/tgorgulu/workspace/tgorgulu/carla_9_15"
    code_path = ""
    assert os.path.exists(carla_path)
    hpc=True


def run_subprocess(name):
    print(name)
    server_process = subprocess.Popen(name, shell=True, preexec_fn=os.setsid)

def run_python_code(name):
    #cmd = "python3 " + "run_screen.py"
    cmd = "python3 " + name
    run_subprocess(cmd)



print(os.getcwd())
if not hpc:
    sing_cmd = "singularity exec --nv --bind /datasets,/workspace,/media,/home,/workspace/tg22/containers /workspace/tg22/containers/leaderboard.sig "
else:
    sing_cmd = "singularity exec --nv --bind /cta/share/tair/,/cta/users/tgorgulu/workspace //cta/users/tgorgulu/containers/leaderboard.sif "

hostname = socket.gethostname()
if hpc:
    result = subprocess.run(["hostname", "-I"], stdout=subprocess.PIPE, universal_newlines=True)
else:
    result = subprocess.run(["hostname", "-I"], stdout=subprocess.PIPE, text=True)
# This returns a string of IP addresses separated by spaces.
ip_address = result.stdout.strip().split()[0]
print("Hostname:", hostname)
print("IP Address:", ip_address)
run_python_code("run_carla.py")
time.sleep(30)
print("run_leaderboard_python","*"*50)
run_subprocess(sing_cmd + "python run_leaderboard_python.py")

time.sleep(10)
while True:
    pass
