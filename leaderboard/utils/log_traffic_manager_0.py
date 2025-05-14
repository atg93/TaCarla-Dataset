import os
import h5py
import numpy as np
import carla
#from leaderboard.autoagents.traditional_agents_files.utils import Run_Route_Log

class Log_Traffic_Manager():

    def __init__(self,run_route_log):
        self.run_route_log = run_route_log
        print("Log_Traffic_Manager:", self.run_route_log.filename)
        self.actor_dics = {}
        self.actor_type_id = {}
        self.read_data(self.run_route_log.filename)
        asd = 0
        self.log_timestep = 0
        self.log_id_to_current_id_dict = {}

    def save_world(self,world):
        self.world = world#actor_list = self.world.get_actors()

    def find_corressponding_index(self,_actor_in_log):
        return np.argmin(np.abs((self.log_timestep) - self.actor_dics[_actor_in_log]['time_step'].flatten()))#np.argwhere(self.log_timestep == self.actor_dics[_actor_in_log]['time_step'].flatten()).flatten()[0]

    def update_current_id_dict(self, ego_loc, ego_vehilc_id):
        self.near_by_vehicle_list = []
        for _actor_in_log in self.actor_dics.keys():
            collect_list = False
            if len(self.near_by_vehicle_list) == 0:
                collect_list = True
            for _actor in self.actor_list:
                if 'vehicle' not in _actor.type_id.split('.') or _actor.id==ego_vehilc_id or _actor_in_log in self.log_id_to_current_id_dict.keys():
                    continue

                location1 = _actor.get_location()
                if location1.distance(ego_loc) < 300 and collect_list:
                    self.near_by_vehicle_list.append(_actor)



                if self.actor_dics[_actor_in_log]['actor_type'] == _actor.type_id and [self.log_timestep] in self.actor_dics[_actor_in_log]['time_step']:

                    index = self.find_corressponding_index(_actor_in_log)
                    new_location = carla.Location(x=self.actor_dics[_actor_in_log]['loc'][index][0], y=self.actor_dics[_actor_in_log]['loc'][index][1],
                                                  z=self.actor_dics[_actor_in_log]['loc'][index][2])

                    if location1.distance(new_location) < 1.0:
                        self.log_id_to_current_id_dict.update({_actor_in_log:_actor})


    def _log_traffic_manager_tick(self,ego_loc,ego_vehilc_id):
        self.actor_list = self.world.get_actors()

        self.update_current_id_dict(ego_loc, ego_vehilc_id)

        self._update_actors()

        self.log_timestep += 1


    def _update_actors(self):
        for log_actor_id in self.log_id_to_current_id_dict.keys():
            if [self.log_timestep] in self.actor_dics[log_actor_id]['time_step'] or True:
                _actor = self.log_id_to_current_id_dict[log_actor_id]
                index = 0#self.find_corressponding_index(log_actor_id)

                log_location = self.actor_dics[log_actor_id]['loc'][index]
                log_rotation = self.actor_dics[log_actor_id]['rot'][index]
                log_speed = self.actor_dics[log_actor_id]['speed'][index]


                new_location = carla.Location(x=log_location[0],y=log_location[1], z=log_location[2])
                new_rotation = carla.Rotation(pitch=log_rotation[0],yaw=log_rotation[1], roll=log_rotation[2])
                new_speed = carla.Vector3D(x=log_speed[0],y=log_speed[1], z=log_speed[2])
                new_transform = carla.Transform(new_location, new_rotation)

                _actor.set_transform(new_transform)
                _actor.set_target_velocity(new_speed)


    def read_data(self, filename):
        with h5py.File(filename, 'r') as file:
            self.current_scenario_name = file['current_scenario_name'][0][0].decode('utf-8')
            for _key in file.keys():
                if 'actor' not in _key.split('-'):
                    continue

                actor_id = _key.split('-')[1]

                if 'vehicle' in _key.split('-')[2].split('.') and actor_id not in self.actor_dics.keys():
                    print("_key:",_key)#file[_key]
                    actor_id = _key.split('-')[1]
                    speed_key = '-'.join(_key.split('-')[:-1]+['speed'])
                    control_key = '-'.join(_key.split('-')[:-1]+['control'])
                    loc_key = '-'.join(_key.split('-')[:-1]+['loc'])
                    rot_key = '-'.join(_key.split('-')[:-1]+['rot'])
                    time_step = '-'.join(_key.split('-')[:-1]+['time_step'])
                    self.actor_dics.update({actor_id:{'speed':np.array(file[speed_key]),'actor_type':_key.split('-')[2],'control':np.array(file[control_key]),'loc':np.array(file[loc_key]),'rot':np.array(file[rot_key]),'time_step':np.array(file[time_step])}})
                elif 'traffic_light' in _key.split('-')[2].split('.') and actor_id not in self.actor_dics.keys():
                    print("_key:",_key)#file[_key]
                    actor_id = _key.split('-')[1]
                    control_key = '-'.join(_key.split('-')[:-1]+['state'])
                    loc_key = '-'.join(_key.split('-')[:-1]+['loc'])
                    rot_key = '-'.join(_key.split('-')[:-1]+['rot'])
                    time_step = '-'.join(_key.split('-')[:-1]+['time_step'])
                    self.actor_dics.update({actor_id:{'actor_type':_key.split('-')[2], 'control':np.array(file[control_key]),'loc':np.array(file[loc_key]),'rot':np.array(file[rot_key]),'time_step':np.array(file[time_step])}})

                asd = 0

    def __call__(self):
        pass