import os.path
import os
import runpy
import socket
import subprocess
import sys
import time


hostname = socket.gethostname()
"""if hostname == "tair-idea":
    ip_address = "10.93.16.131"
elif hostname == "tair-helix":
    ip_address = """""

try:
    result = subprocess.run(["hostname", "-I"], stdout=subprocess.PIPE, text=True)
except:
    result = subprocess.run(["hostname", "-I"], stdout=subprocess.PIPE, universal_newlines=True)

# This returns a string of IP addresses separated by spaces.
ip_address = result.stdout.strip().split()[0]
print("Hostname:", hostname)
print("IP Address:", ip_address)


def is_port_in_use(port, host):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Try to bind to the host and port
        s.bind((host, port))
    except socket.error:
        # If binding fails, the port is in use
        return True
    else:
        # Port is free; close the socket immediately
        s.close()
        return False


try:
    carla_path = "//workspace/tg22/carla_9_15/"
    code_path = ""
    assert os.path.exists(carla_path)
    hpc=False
    carla_command = "/workspace/tg22/carla_9_15/CarlaUE4.sh -RenderOffScreen"
    singurality_cmd = "singularity run --nv --bind /datasets,/workspace,/media //workspace/tg22/containers_yedek/carla_0.9.15.sif"
except:
    carla_path = "//cta/users/tgorgulu/workspace/tgorgulu/carla_9_15"
    code_path = ""
    assert os.path.exists(carla_path)
    hpc=True
    sys.path.insert(0, "//cta/users/tgorgulu/")
    carla_command = "/cta/users/tgorgulu/workspace/tgorgulu/carla_9_15/CarlaUE4.sh -RenderOffScreen"
    singurality_cmd = "singularity run --nv --bind  /cta/share/tair/,/cta/users/tgorgulu/workspace  //cta/users/tgorgulu/containers_local/carla_15.sif"

if hpc:
    port_list = [2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070]
    sys.path.insert(0, "/cta")
else:
    port_list = [2000, 2010, 2020, 2030]
    #port_list = [2050]


sys.path.insert(0, carla_path)
process_list = []

while True:
    for index, port in enumerate(port_list):
        in_use = is_port_in_use(port=port, host=ip_address)
        if not in_use:
            print("port:", port)
            time.sleep(10)
            if not hpc:
                carla_process = subprocess.Popen(
                    [singurality_cmd + " ", "."+carla_command,
                    '-graphicsadapter='+str(index),
                    '-nosound',
                    '-carla-rpc-port='+ str(port)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,  # This makes stdout/stderr strings instead of bytes
                    shell=True,
                    executable="/bin/bash"
                )
            else:
                carla_process = subprocess.Popen(
                    [singurality_cmd + " ", "."+carla_command,
                    '-graphicsadapter='+str(index),
                    '-nosound',
                    '-carla-rpc-port='+ str(port)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,  # This makes stdout/stderr strings instead of bytes
                    shell=True,
                    executable="/bin/bash"
                )
            cmd = singurality_cmd + "  /"+carla_command+" -graphicsadapter="+str(index)+" -nosound -carla-rpc-port=" + str(port)
            print(cmd)
            # log_file = self._root_save_dir / f'server_{cfg["port"]}.log'
            # server_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid, stdout=open(log_file, "w"))
            server_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
            process_list.append(server_process)
            time.sleep(10)
