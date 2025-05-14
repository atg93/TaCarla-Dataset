import subprocess


def create_screen_session(session_name, command):
    try:
        # Create a new screen session with the specified name
        subprocess.run(['screen', '-S', session_name, '-dm'], check=True)

        # Send the command to the screen session
        subprocess.run(['screen', '-S', session_name, '-X', 'stuff', f'{command}\n'], check=True)

        print(f"Screen session '{session_name}' created and command '{command}' started.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


print("asd")
# Example usage
#create_screen_session('my_session', 'echo "Hello, World!"')