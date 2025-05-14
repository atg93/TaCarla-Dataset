pickle_file_path = "/workspace/tg22/leaderboard_data/detection_dataset/episode_14_pickle_file/info.pickle"
#path = "/workspace/tg22/leaderboard_data/info.pickle"


import pickle

# Specify the path to your pickle file
data = []
with open(pickle_file_path, 'rb') as file:
    while True:
        try:
            # Attempt to load the next object in the file
            object = pickle.load(file)
            data.append(object)
        except EOFError:
            # End of file reached
            break

# Now you can use the loaded data
print(data)

