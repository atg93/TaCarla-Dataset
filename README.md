The aim of this code is to demonstrate how the TaCarla dataset was collected for the Autonomous Driving Data Collection in the Carla Leaderboard 2.0 Challenge.
This repository is designed to collect data samples for autonomous driving scenarios in the CARLA Leaderboard 2.0 challenge. The data collected from various autonomous driving simulations is intended to aid in the development, testing, and evaluation of AI models for real-world autonomous driving applications.

Repository Overview
This repository contains scripts and tools for:

Data collection from CARLA simulation

Visualizing the Collected TaCarla Dataset

Organizing and storing data in a structure suitable for training autonomous driving models

The collected dataset includes but is not limited to:

Dynamic object detection

Lane detection

Planning and decision-making tasks

Traffic light and pedestrian crossing scenarios

Ego vehicle control data

Setup and Installation
Prerequisites
# Before using this repository:

# CARLA: A high-fidelity simulator for autonomous driving research. 

To download CARLA 0.9.15, please visit: https://github.com/carla-simulator/carla/releases . 

For running the CARLA container, download the "singularity build carla_0.9.15.sif docker-archive://carla_0.9.15.tar" 

To run Carla simulation please run this command: 

singularity run --nv --bind /datasets,/workspace,/media *CONTAINER_PATH* *CARLA_PATH* -RenderOffScreen -graphicsadapter=0 -nosound -carla-rpc-port=2000

CONTAINER_PATH should be like: //home/tg22/containers/carla_15.sif

CARLA_PATH should be like: ./home/tg22/carla_9_15/CarlaUE4.sh

# Running Python, 

To run the code, you can simply create a virtual environment with the following commands:

Required Python libraries in requirements.txt

To run the code, you can simply create a virtual environment with the following commands:

conda create --name tacarla python=3.8

source activate tacarla

pip install -r requirements.txt

or 

Alternatively, you can use the Singularity container from here:

https://drive.google.com/file/d/1WEUG6WvVbzT1tB6IVhmmvjHbqvDF-sPK/view?usp=drive_link



# Collecting Data

The parameters in the main function of leaderboard_evaluator.py—'TOWN_NAME', 'code_path', 'DATASAVEPATH', and 'TEAMCODE_PATH'—need to be arranged properly for the code to run correctly. In a later release, these parameters will be added to config.yaml.

To begin collecting data, please run:

python TaCarla_Dataset/leaderboard/leaderboard_evaluator.py --agent=/leaderboard/autoagents/traditional_agents_0.py --port=2000 --traffic-manager-port=6000 --debug=0 --track=MAP --record=1 --routes=SCENARIO_NAME

SCENARIO_NAME should be like: train_data_trigger_point/SignalizedJunctionLeftTurn_1.xml
