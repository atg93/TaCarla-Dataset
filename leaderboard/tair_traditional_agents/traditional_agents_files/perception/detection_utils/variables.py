
import os
from future_prediction.bev.control.multiple_future_prediction import Multiple_Future_Prediction
import torch


class Variables:

    def __init__(self, cfg, args, episode_number):

        self.variable_dict = {
            "log_wandb": cfg['obs_configs']['birdview']['general']['log_wandb'],
            "leaderboard_version": cfg['obs_configs']['birdview']['general']['leaderboard_version'],
            "model_path": args.model_path,
            "episode_number": episode_number,
            "use_future_prediction": cfg['obs_configs']['birdview']['general']['use_future_prediction'],
            "log_dataset": cfg['obs_configs']['birdview']['general']['log_dataset'],
            "width": cfg['obs_configs']['birdview']['width_in_pixels'],
            "video_dir": cfg['obs_configs']['birdview']['general']['video_dir'] + str(episode_number) + '/',
            "save_video": cfg['obs_configs']['birdview']['general']['save_video'],
            "human_mode": cfg['obs_configs']['birdview']['general']['human_mode'],
            "speed_boost": cfg['obs_configs']['birdview']['demo']['speed_boost'],
            "manual_speed": cfg['obs_configs']['birdview']['demo']['manual_speed'],
            "wheel_control": cfg['obs_configs']['birdview']['demo']['wheel_control'],
            "countdown_seconds": cfg['obs_configs']['birdview']['demo']['countdown_seconds'],
            "wheel_turning_threshold": cfg['obs_configs']['birdview']['demo']['wheel_turning_threshold'],
            "use_perception": cfg['obs_configs']['birdview']['general']['use_perception'],
            "monocular_perception": cfg['obs_configs']['birdview']['perception']['monocular'],
            "perception_checkpoint": cfg['obs_configs']['birdview']['perception']['checkpoint'],
            "history_idx": cfg['obs_configs']['birdview']['history_idx'],
            "use_opendrive": cfg['obs_configs']['birdview']['general']['use_opendrive']

        }

        if self.variable_dict["log_dataset"]:
            self.variable_dict["log_nuplan"] = cfg['obs_configs']['birdview']['log_dataset']['log_nuplan']
            self.variable_dict["nuplan_dataset_path"] = cfg['obs_configs']['birdview']['log_dataset']['nuplan_dataset_path']
            if self.variable_dict["log_nuplan"]:
                if not os.path.exists(self.variable_dict["nuplan_dataset_path"]):
                    os.makedirs(self.variable_dict["nuplan_dataset_path"])
            self.variable_dict["log_detection"] = cfg['obs_configs']['birdview']['log_dataset']['log_detection']
            self.variable_dict["detection_dataset_path"] = cfg['obs_configs']['birdview']['log_dataset']['detection_dataset_path'] + \
                                          'episode_' + str(episode_number) + '/'
            self.variable_dict["detection_images_path"] = self.variable_dict["detection_dataset_path"] + 'images/'
            self.variable_dict["cams"] = ['front_left', 'front', 'front_right', 'back_left', 'back', 'back_right']
            if self.variable_dict["log_detection"]:
                if not os.path.exists(self.variable_dict["detection_dataset_path"]):
                    os.makedirs(self.variable_dict["detection_dataset_path"])
                for i in range(len(self.variable_dict["cams"])):
                    if not os.path.exists(self.variable_dict["detection_images_path"] + self.variable_dict["cams"][i] + '/'):
                        os.makedirs(self.variable_dict["detection_images_path"] + self.variable_dict["cams"][i] + '/')
                self.variable_dict["img_counter"] = 0
            self.variable_dict["time_window"] = cfg['obs_configs']['birdview']['log_dataset']['time_window']
        else:
            self.variable_dict["log_detection"] = False
            self.variable_dict["log_nuplan"] = False

        if self.variable_dict["use_future_prediction"]:
            checkpoint_path = cfg['obs_configs']['birdview']['future_prediction']['checkpoint']
            device = torch.device('cuda:2')
            print("!" * 100)
            print("set_future_prediction is created")
            future_prediction = Multiple_Future_Prediction(checkpoint_path=checkpoint_path, dataroot=None,
                                                           device=device)
            self.variable_dict["future_prediction"] = future_prediction
            self.variable_dict["break_counter"] = cfg['obs_configs']['birdview']['future_prediction']['break_counter']

        if self.variable_dict["save_video"]:
            if not os.path.exists(self.variable_dict["video_dir"]):
                os.makedirs(self.variable_dict["video_dir"])

        if self.variable_dict["wheel_control"]:
            self.variable_dict["force_feedback"] = cfg['obs_configs']['birdview']['demo']['force_feedback']
        else:
            self.variable_dict["force_feedback"] = False

        if self.variable_dict["use_perception"]:
            self.variable_dict["lane_detection"] = cfg['obs_configs']['birdview']['perception']['lane_detection']
            if self.variable_dict["lane_detection"]:
                self.variable_dict["lane_yaml_path"] = cfg['obs_configs']['birdview']['perception']['lane_yaml_path']


