import logging
import os

import numpy as np
from omegaconf import OmegaConf
import wandb
import copy
import pickle

from carla_gym.utils.config_utils import load_entry_point

#from agents_roach.rl_birdview.models.sac.sac import SAC


class RlBirdviewAgent():
    def __init__(self, path_to_conf_file='config_agent.yaml',benchmark=False,lane_benchmark=False,pretrained=False):
        self._logger = logging.getLogger(__name__)
        self._render_dict = None
        self.supervision_dict = None
        self.load_path = '/workspace/tg22/roach_yedek/roach/carla-roach/outputs' + '/test_model/'
        self.setup(path_to_conf_file,benchmark,lane_benchmark,pretrained)
        self.freeze_weighs = True


    def setup(self, path_to_conf_file, benchmark, lane_benchmark, pretrained):
        cfg = OmegaConf.load(path_to_conf_file)
        # load checkpoint from wandb
        self._ckpt = None
        self.pretrained = pretrained
        cfg = OmegaConf.to_container(cfg)


        if benchmark == False:
            self._obs_configs = cfg['obs_configs']
            self._train_cfg = cfg['training']
            self._policy = None
            if pretrained:
                file_name = 'imitation_model.pickle'
                print("#" * 50)
                print("Pretrained_model " + file_name + " is loaded")
                load_path = '/home/tg22/remote-pycharm/roach/carla-roach/outputs' + '/pretrained_model/'
                with open(load_path + file_name, 'rb') as pickle_file:
                    self._policy = pickle.load(pickle_file)
                self._policy.update_device_num(device_num=1)
                self._policy = self._policy.train()
        else:
            # prepare policy
            # for original pretrained roach policy
            with open(self.load_path + 'agent.pickle', 'rb') as pickle_file:
                agents_pickle = pickle.load(pickle_file)
            self._obs_configs = agents_pickle.obs_configs


            cfg = self.correction_for_cfg(cfg)

            #self.load_path = '/home/tg22/remote-pycharm/roach/carla-roach/outputs' + '/test_model/'
            #'ppo_policy800_new_size.pickle'ppo_policy800
            #'imitation_model.pickle'
            with open(self.load_path+'ppo_policy400.pickle','rb') as pickle_file:
               self._policy = pickle.load(pickle_file)
            #self._policy = agents_pickle._policy
            #self._policy.update_device_num(device_num=1)

            for param in self._policy.parameters():
                param.requires_grad = False

            if lane_benchmark:
                self._policy = agents_pickle._policy
                self._policy.update_device_num(device_num=0)
            self._policy = self._policy.eval()

        #cfg = OmegaConf.to_container(cfg)

        #self._obs_configs = cfg['obs_configs']
        #self._train_cfg = cfg['training']

        # prepare policy
        self._policy_class = load_entry_point(cfg['policy']['entry_point'])
        self._policy_kwargs = cfg['policy']['kwargs']

        self._wrapper_class = load_entry_point(cfg['env_wrapper']['entry_point'])
        self._wrapper_kwargs = cfg['env_wrapper']['kwargs']

    def run_step(self, input_data, timestamp=0, prediction_output=None, e_brake=False):
        input_data = copy.deepcopy(input_data)
        policy_input = self._wrapper_class.process_obs(input_data, self._wrapper_kwargs['input_states'], train=False)
        self.policy_input = policy_input

        actions, values, log_probs, mu, sigma, features = self._policy.forward(
            policy_input, deterministic=True, clip_action=True)


        control = self._wrapper_class.process_act(actions, self._wrapper_kwargs['acc_as_action'], train=False)

        return control

    def get_attention_map(self):
        return self._policy.get_attention_map()

    def run_for_e_brake(self, input_data):
        actions, values, log_probs, mu, sigma, features = self._policy.forward(
            input_data, deterministic=True, clip_action=True)
        return actions

    def reset(self, log_file_path):
        # logger
        self._logger.handlers = []
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(logging.DEBUG)
        self._logger.addHandler(fh)

    def learn(self, env, total_timesteps, callback, seed):
        print("self._policy_class:",self._policy_class)
        if self._policy is None:
            self._policy = self._policy_class(env.observation_space, env.action_space, **self._policy_kwargs)
            self._policy.update_device_num(device_num=1)

        # init ppo model
        model_class = load_entry_point(self._train_cfg['entry_point'])
        model = model_class(self._policy, env, **self._train_cfg['kwargs'])
        #model_class = load_entry_point("agents_roach.rl_birdview.models.sac.sac.sac:SAC")
        #model = model_class(self._policy, env)
        model.learn(total_timesteps, callback=callback, seed=seed)

    def render(self, reward_debug, terminal_debug):
        '''
        test render, used in benchmark.py
        '''
        self._render_dict['reward_debug'] = reward_debug
        self._render_dict['terminal_debug'] = terminal_debug

        return self._wrapper_class.im_render(self._render_dict)

    @property
    def obs_configs(self):
        return self._obs_configs

    def render(self, reward_debug,im_birdview):
        '''
        test render, used in benchmark.py
        '''
        self._render_dict['reward_debug'] = reward_debug
        #self._render_dict['terminal_debug'] = terminal_debug

        return self._wrapper_class.im_render(self._render_dict,im_birdview)

    def correction_for_cfg(self,cfg):
        cfg['entry_point'] = "agents_roach.rl_birdview.rl_birdview_agent:RlBirdviewAgent"
        cfg['policy'] = {}
        cfg['policy']['entry_point'] = "agents_roach.rl_birdview.models.ppo_policy:PpoPolicy"
        cfg['policy']['kwargs'] = {}
        cfg['policy']['kwargs'][
            'features_extractor_entry_point'] = "agents_roach.rl_birdview.models.torch_layers:XtMaCNN"
        cfg['policy']['kwargs'][
            'distribution_entry_point'] = "agents_roach.rl_birdview.models.distributions:BetaDistribution"
        cfg['env_wrapper']['entry_point'] = "agents_roach.rl_birdview.utils.rl_birdview_wrapper:RlBirdviewWrapper"
        cfg['training'] = {}
        cfg['training']['entry_point'] = "agents_roach.rl_birdview.models.ppo:PPO"
        self._obs_configs['birdview']['width_in_pixels'] = 200
        self._obs_configs['birdview']['pixels_ev_to_bottom'] = 100
        self._obs_configs['birdview']['pixels_per_meter'] = 4
        return cfg

    def set_policy(self,policy):
        self._policy = policy

    def get_policy_input(self):
        return self.policy_input

