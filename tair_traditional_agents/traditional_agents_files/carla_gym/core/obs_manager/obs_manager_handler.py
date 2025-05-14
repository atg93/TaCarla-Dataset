from importlib import import_module
from gym import spaces

class ObsManagerHandler(object):

    def __init__(self, obs_configs, lane_model_available=False,future_prediction_dataset_label=False):
        self.lane_model_available = lane_model_available
        self.future_prediction_dataset_label = future_prediction_dataset_label
        self._obs_managers = {}
        self._obs_configs = obs_configs
        self._init_obs_managers()
        self.wo_hd = True

    def get_observation(self, timestamp):
        obs_dict = {}
        for ev_id, om_dict in self._obs_managers.items():
            obs_dict[ev_id] = {}
            for obs_id, om in om_dict.items():
                obs_dict[ev_id][obs_id] = om.get_observation()
        return obs_dict

    def get_is_there_list_parameter(self):
        for ev_id, om_dict in self._obs_managers.items():
            return om_dict['birdview'].get_is_there_list_parameter()

    def set_control_with(self, _state):
        for ev_id, om_dict in self._obs_managers.items():
            om_dict['birdview'].set_control_with(_state)

    def set_sensors(self,*arg):
        for ev_id, om_dict in self._obs_managers.items():
            om_dict['birdview'].set_sensor(arg)

    def get_plant_control(self):
        for ev_id, om_dict in self._obs_managers.items():
            return om_dict['birdview'].get_plant_control()




    def set_trafic_light_disable(self, trafic_light_disable):
        for ev_id, om_dict in self._obs_managers.items():
            om_dict['birdview'].set_trafic_light_disable(trafic_light_disable)


    def set_future_prediction_obj(self, timestamp, future_prediction):
        obs_dict = {}
        for ev_id, om_dict in self._obs_managers.items():
            om_dict['birdview'].set_future_prediction_obj(future_prediction)

    def is_there_intersection(self,prediction_output):
        for ev_id, om_dict in self._obs_managers.items():
            return om_dict['birdview'].is_there_intersection(prediction_output)
        

    def collect_data_for_future_prediction(self, timestamp):
        obs_dict = {}
        for ev_id, om_dict in self._obs_managers.items():
            om_dict['birdview'].collect_data_for_future_prediction()

    def reset_episode(self, timestamp):
        obs_dict = {}
        for ev_id, om_dict in self._obs_managers.items():
            om_dict['birdview'].reset_episode()

    def set_episode_number(self, task_idx):
        obs_dict = {}
        for ev_id, om_dict in self._obs_managers.items():
            om_dict['birdview'].set_episode_number(task_idx)

    def close_h5_file(self):
        obs_dict = {}
        for ev_id, om_dict in self._obs_managers.items():
            om_dict['birdview'].close_h5_file()

    def set_collect_data_info(self,data_collection):
        for ev_id, om_dict in self._obs_managers.items():
            om_dict['birdview'].set_collect_data_info(data_collection)

    def get_ego_and_wp_list(self):
        for ev_id, om_dict in self._obs_managers.items():
            output = om_dict['birdview'].get_ego_and_wp_list()
        return output

    def get_ego_transform(self):
        for ev_id, om_dict in self._obs_managers.items():
            output = om_dict['birdview'].get_ego_transform()
        return output

    def get_current_route_plan(self):
        for ev_id, om_dict in self._obs_managers.items():
            output = om_dict['birdview'].get_current_route_plan()
        return output

    def update_route_plan(self,new_route):
        for ev_id, om_dict in self._obs_managers.items():
            om_dict['birdview'].update_route_plan(new_route)

    def set_orginal_route_label(self,status=True):
        for ev_id, om_dict in self._obs_managers.items():
            om_dict['birdview'].set_orginal_route_label(status)
    
    def get_wp_points(self):
        for ev_id, om_dict in self._obs_managers.items():
            return om_dict['birdview'].get_wp_points()

    def set_sensors_interface(self,interface):
        for ev_id, om_dict in self._obs_managers.items():
            om_dict['birdview'].set_sensors_interface(interface)

    @property
    def observation_space(self):
        obs_spaces_dict = {}
        for ev_id, om_dict in self._obs_managers.items():
            ev_obs_spaces_dict = {}
            for obs_id, om in om_dict.items():
                ev_obs_spaces_dict[obs_id] = om.obs_space
            obs_spaces_dict[ev_id] = spaces.Dict(ev_obs_spaces_dict)
        return spaces.Dict(obs_spaces_dict)

    def reset(self, ego_vehicles):
        self._init_obs_managers()

        for ev_id, ev_actor in ego_vehicles.items():
            for obs_id, om in self._obs_managers[ev_id].items():
                om.attach_ego_vehicle(ev_actor)

    def clean(self):
        for ev_id, om_dict in self._obs_managers.items():
            for obs_id, om in om_dict.items():
                om.clean()
        self._obs_managers = {}

    def _init_obs_managers(self):
        for ev_id, ev_obs_configs in self._obs_configs.items():
            self._obs_managers[ev_id] = {}
            for obs_id, obs_config in ev_obs_configs.items():
                if self.lane_model_available and obs_config["module"] == "birdview.chauffeurnet":
                    ObsManager = getattr(import_module('carla_gym.core.obs_manager.' + "birdview.chauffeurnet_lane_model"),
                                         'ObsManager')
                    self._obs_managers[ev_id][obs_id] = ObsManager(obs_config)
                else:
                    ObsManager = getattr(import_module('carla_gym.core.obs_manager.' + obs_config["module"]),
                                         'ObsManager')
                    self._obs_managers[ev_id][obs_id] = ObsManager(obs_config)

                if self.future_prediction_dataset_label and obs_config["module"] == "birdview.chauffeurnet":

                    ObsManager = getattr(
                        import_module('carla_gym.core.obs_manager.' + "birdview.chauffeurnet_datasets"),
                        'ObsManager')
                    self._obs_managers[ev_id]['birdview_future_datasets'] = ObsManager(obs_config)

                if obs_config["module"] == "birdview.chauffeurnet":
                    ObsManager = getattr(
                        import_module('carla_gym.core.obs_manager.' + "birdview.chauffeurnet_wo_hd"),
                        'ObsManager')
                    self._obs_managers[ev_id][obs_id] = ObsManager(obs_config)


    def check_brake_label(self, prediction_output):
        for ev_id, om_dict in self._obs_managers.items():
            return om_dict['birdview'].check_brake_label(prediction_output)
