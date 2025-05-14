import subprocess

def create_screen_session(session_name, command_to_run):
    # Start a detached screen session
    subprocess.run(["screen", "-dmS", session_name])
    # Run the command in the screen session
    subprocess.run(["screen", "-S", session_name, "-X", "stuff", f"{command_to_run}\n"])

# Define the number of screen sessions you want to create
num_sessions = 5
path_to_python_script = "/path/to/script/my_script.py"

# Loop through and create each screen session, running a Python script in each
for i in range(num_sessions):
    session_name = f"session_{i}"
    command_to_run = "ls"#f"python {path_to_python_script}"
    create_screen_session(session_name, command_to_run)