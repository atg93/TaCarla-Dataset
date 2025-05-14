import os.path
import os
import socket
import subprocess
import sys
import time
import psutil
import os
import signal
from selecting_task_for_datacollection import seleting_task

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

if hpc:
    port_list = [2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070]
else:
    port_list = [2000, 2010, 2020, 2030]

hostname = socket.gethostname()
result = subprocess.run(["hostname", "-I"], stdout=subprocess.PIPE, text=True)
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


def get_process_name(port):
    # Iterate over all network connections
    process_list = []
    pid_list = []
    for conn in psutil.net_connections():
        # Ensure the local address exists and check its port
        if conn.laddr and conn.laddr.port == port:
            pid = conn.pid
            if pid:
                proc = psutil.Process(pid)
                pid_list.append(pid)
                process_list.append(proc.name())

    return process_list, pid_list

def kill_pid(pid):
    try:
        os.kill(pid, signal.SIGTERM)  # Sends SIGTERM to request graceful termination
        print(f"Process {pid} has been signaled to terminate.")
    except ProcessLookupError:
        print(f"No process found with PID {pid}.")
    except PermissionError:
        print("Insufficient permissions to kill the process.")

#'CarlaUE4-Linux-Shipping'
if hpc:
    sing_cmd = "singularity exec --nv --bind /cta/share/tair/,/cta/users/tgorgulu/workspace //cta/users/tgorgulu/containers/leaderboard.sif "
    carla_path = "//cta/users/tgorgulu/workspace/tgorgulu/carla_9_15"
else:
    sing_cmd = "singularity exec --nv --bind /datasets,/workspace,/media //home/tg22/containers/leaderboard.sig "
    carla_path = "//workspace/tg22/carla_9_15/"

code_path = ""
assert os.path.exists(carla_path)
sys.path.insert(0, carla_path)
process_list = []
process_dict = {}
global_selected_task_list = []

if os.getenv("TOWN_NAME") == "Town12":
    route_path = "train_data_trigger_point/"  # tugrul_map

elif os.getenv("TOWN_NAME") == "Town13":
    route_path = "val_data_trigger_point/" #tugrul_map

print("TOWN_NAME: ", os.getenv("TOWN_NAME"))
print("route_path: ",route_path)
while True:
    for index, port in enumerate(port_list):
        in_use = is_port_in_use(port=port, host=ip_address)
        if port not in list(process_dict.keys()):
            _, pid_list = get_process_name(port)
            for _ in range(10):
                for _pid in pid_list:
                    kill_pid(_pid)
            time.sleep(60)
            print("leaderboard port: ",port)
            selected_task = seleting_task(global_selected_task_list) #"Accident_1"
            process = subprocess.Popen([
                'python', 'leaderboard/leaderboard_evaluator.py',
                '--agent=/leaderboard/autoagents/traditional_agents_0.py',
                '--port='+str(port),
                '--traffic-manager-port='+ str(4000 + port),
                '--debug=0',
                '--track=MAP',
                '--record=1',
                '--routes=' + route_path +selected_task #+'.xml'
            ])
            process_dict.update({port:process})
            time.sleep(15)

        try:
            retcode = process_dict[port].poll()  # None means it's still running
        except:
            retcode = 1

        if retcode is not None:
            print(f"Process finished with exit code: {retcode}")
            del process_dict[port]
            asd = 0
        else:
            time.sleep(1)

            asd = 0
