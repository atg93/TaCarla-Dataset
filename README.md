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
Before using this repository, ensure that the following dependencies are installed:

CARLA: A high-fidelity simulator for autonomous driving research. To download CARLA 0.9.15, please visit: https://github.com/carla-simulator/carla/releases . For running the CARLA container, download the "singularity build carla_0.9.15.sif docker-archive://carla_0.9.15.tar" 


Python 3.x: Python programming language (preferably Python 3.7 or later).

Required Python libraries in requirements.txt

The parameters in the main function of leaderboard_evaluator.py—'TOWN_NAME', 'code_path', 'DATASAVEPATH', and 'TEAMCODE_PATH'—need to be arranged properly for the code to run correctly. In a later release, these parameters will be added to config.yaml.
