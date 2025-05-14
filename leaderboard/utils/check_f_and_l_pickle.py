import pickle

pickle_file_path = 'episode_21.pickle'
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

type_list = []
for index in range(len(data)):
    sample = data[index][2]
    if len(sample) != 0:
        for s in sample:
            type_list.append(s['name'])
        asd = 0
    asd = 0

print(set(type_list))
asd = 0