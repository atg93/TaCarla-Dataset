import os
import shutil


if Town_name=="Town13":
    try:
        directory_path = '/workspace/tg22/leaderboard_plant_pdm/'
        destination_directory_path = '/workspace/tg22/leaderboard_plant_pdm/checked/'
        assert os.path.exists(directory_path)
    except:
        directory_path = "/cta/users/tgorgulu/workspace/tgorgulu/leaderboard_plant_pdm/"
        destination_directory_path = '/cta/users/tgorgulu/workspace/tgorgulu/leaderboard_plant_pdm/checked/'
        assert os.path.exists(directory_path)

folders = os.listdir(directory_path)

for fol in folders:
    if 'score.txt' in os.listdir(directory_path+fol):
        score_path = directory_path + fol + '/' + 'score.txt'

        with open(score_path, 'r') as file:
            content = file.read()

        if float(content.split('score_composed:')[1].split(' ')[1]) > 70.0: #float(content.split('score_composed:')[1].split(' ')[1]) > 70.0: #score_penalty
            source = directory_path + fol
            destination = destination_directory_path + fol

            # Move the folder
            shutil.move(source, destination)

        asd = 0