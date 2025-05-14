import pickle
import sys
from omegaconf import DictConfig, OmegaConf

from leaderboard.autoagents.traditional_agents_files.roach.carla_gym.utils import config_utils


sys.path.append(
    '/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/roach/')

#/cta/users/tgorgulu/git/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/
sys.path.append(
    '/cta/users/tgorgulu/git/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/roach/')

from carla_gym.utils import config_utils
from carla_gym.core.obs_manager.actor_state.velocity import ObsManager
import numpy as np

class Run_Roach():
    def __init__(self, parent_actor):
        path = '/home/tg22/remote-pycharm/leaderboard2.0/leaderboard/autoagents/traditional_agents_files/roach/' \
               'roach_cfg.pickle'

        with open(path,'rb') as pickle_file:
            cfg = pickle.load(pickle_file)

        agents_dict = {}
        obs_configs = {}
        reward_configs = {}
        terminal_configs = {}
        agent_names = []
        for ev_id, ev_cfg in cfg.actors.items():
            agent_names.append(ev_cfg.agent)
            cfg_agent = cfg.agent[ev_cfg.agent]
            OmegaConf.save(config=cfg_agent, f='config_agent.yaml')
            self.AgentClass = config_utils.load_entry_point(cfg_agent.entry_point)
            agents_dict[ev_id] = self.AgentClass('config_agent.yaml', benchmark=True)
            obs_configs[ev_id] = agents_dict[ev_id].obs_configs

            # get obs_configs from agent
            reward_configs[ev_id] = OmegaConf.to_container(ev_cfg.reward)
            terminal_configs[ev_id] = OmegaConf.to_container(ev_cfg.terminal)

        self.agents_dict = agents_dict
        self.vel_obs = ObsManager(obs_configs)
        self.vel_obs.attach_ego_vehicle(parent_actor)


        asd = 0


    def __call__(self, input, prev_control):
        vel = self.vel_obs.get_observation()

        input.update({'velocity': vel})
        input.update({'control':{'throttle':np.array([prev_control.throttle]), 'steer':np.array([prev_control.steer]), 'brake':np.array([prev_control.brake]), 'gear':np.array([prev_control.gear])}})

        return self.agents_dict['hero'].run_step(input)