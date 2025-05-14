#!/bin/bash


python3 leaderboard/leaderboard_evaluator.py --agent='/leaderboard/autoagents/human_agent.py' --port=2000 --traffic-manager-port=6000 --routes=data/routes_training.xml --checkpoint="/workspace/leaderboard/results/results.json" --debug=0

