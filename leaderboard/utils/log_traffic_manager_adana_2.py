import os
import h5py
import numpy as np
import carla
# from leaderboard.autoagents.traditional_agents_files.utils import Run_Route_Log
import time
import random


class Log_Traffic_Manager():

    def __init__(self, run_route_log, world):
        self.run_route_log = run_route_log
        self.world = world
        self.created_loc_list = []
        print("Log_Traffic_Manager:", self.run_route_log.filename)
        self.first_creation = True

    def _reset(self):
        self.actor_dics = {}
        self.created_actor_dics = {}
        self.actor_type_id = {}
        self.read_data(self.run_route_log.filename)
        asd = 0
        self.log_timestep = 0
        self.log_id_to_current_id_dict = {}
        self.new_actor_list = []
        self.new_actor_loc_dict = {}

        self.traffic_actor_dics = {}
        self.new_traffic_actor_loc_dict = {}
        self.traffic_log_id_to_actor_id = {}

    def save_world(self, world):
        self.world = world  # actor_list = self.world.get_actors()
        self._reset()

    def find_corressponding_index(self, _actor_in_log):
        return np.argmin(np.abs((self.log_timestep) - self.actor_dics[_actor_in_log][
            'time_step'].flatten()))  # np.argwhere(self.log_timestep == self.actor_dics[_actor_in_log]['time_step'].flatten()).flatten()[0]

    def update_current_id_dict(self, ego_loc, ego_vehilc_id):
        # print("len(self.log_id_to_current_id_dict):",len(self.log_id_to_current_id_dict))
        for _actor_id in self.new_id_list:
            loc_dist_dict = {}
            for _actor_in_log in self.created_actor_dics.keys():

                if _actor_in_log in self.log_id_to_current_id_dict.keys():
                    continue

                _actor = self.world.get_actor(_actor_id)

                location1 = self.new_actor_loc_dict[_actor_id]

                if self.created_actor_dics[_actor_in_log]['actor_type'] == _actor.type_id:
                    index = self.find_corressponding_index(_actor_in_log)
                    new_location = carla.Location(x=self.created_actor_dics[_actor_in_log]['loc'][index][0],
                                                  y=self.created_actor_dics[_actor_in_log]['loc'][index][1],
                                                  z=self.created_actor_dics[_actor_in_log]['loc'][index][2])

                    distance = location1.distance(new_location)
                    loc_dist_dict.update({_actor_in_log: distance})

                    # if distance < 1.0:
                    #    self.log_id_to_current_id_dict.update({_actor_in_log:_actor_id})
            if len(list(loc_dist_dict.keys())) != 0:
                new_id = list(loc_dist_dict.keys())[np.argmin(list(loc_dist_dict.values()))]
                self.log_id_to_current_id_dict.update({new_id: _actor_id})
            asd = 0
        asd = 0  # self.new_actor_loc_dict[_actor_id], self.created_actor_dics[_actor_in_log]['loc']

    def _log_traffic_manager_tick(self, ego_loc, ego_vehilc_id):
        self.actor_list = self.world.get_actors()
        self.create_vehicle(ego_vehilc_id)
        self.destroy_vehicle(ego_vehilc_id)

        for _actor in self.actor_list:
            if 'vehicle' in _actor.type_id.split(
                    '.') and _actor.id not in self.new_actor_list and _actor.id != ego_vehilc_id:
                print("_actor.id:", _actor.id)
                self.new_actor_list.append(_actor.id)
                self.new_actor_loc_dict.update({_actor.id: _actor.get_location()})

        self.new_id_list = []
        for log_id in self.new_actor_loc_dict.keys():
            if log_id not in self.log_id_to_current_id_dict.values():
                print("log_id:", log_id)
                self.new_id_list.append(log_id)

        self.update_current_id_dict(ego_loc, ego_vehilc_id)

        self.update_traffic_lights()

        self._update_actors()

        self.log_timestep += 1

    def find_traffic_corressponding_index(self,_actor_in_log):
        return np.argmin(np.abs((self.log_timestep) - self.traffic_actor_dics[_actor_in_log]['time_step'].flatten()))

    def update_traffic_lights(self):
        #elf.new_traffic_actor_loc_dict
        if len(self.traffic_log_id_to_actor_id) != len(self.traffic_actor_dics):
            all_actors = self.world.get_actors()
            for log_actor_id in self.traffic_actor_dics.keys():
                index = 0
                new_location = carla.Location(x=self.traffic_actor_dics[log_actor_id]['loc'][index][0], y=self.traffic_actor_dics[log_actor_id]['loc'][index][1],
                                              z=self.traffic_actor_dics[log_actor_id]['loc'][index][2])
                loc_dist_dict = {}
                for actor in all_actors:
                    if 'traffic' in actor.type_id.split('.'):
                        location1 = actor.get_location()
                        distance = location1.distance(new_location)
                        loc_dist_dict.update({actor.id:distance})

                self.traffic_log_id_to_actor_id.update({log_actor_id:list(loc_dist_dict.keys())[np.argmin(list(loc_dist_dict.values()))]})

        for log_actor_id in self.traffic_log_id_to_actor_id.keys():
            if [self.log_timestep] in self.traffic_actor_dics[log_actor_id]['time_step']:
                _actor_id = self.traffic_log_id_to_actor_id[log_actor_id]
                _actor = self.world.get_actor(_actor_id)
                index = self.find_traffic_corressponding_index(log_actor_id)

                state = self.traffic_actor_dics[log_actor_id]['control'][index][0].decode('utf-8')

                if state == 'Red':
                    state = carla.TrafficLightState.Red
                elif state == 'Yellow':
                    state = carla.TrafficLightState.Yellow
                elif state == 'Green':
                    state = carla.TrafficLightState.Green

                _actor.set_state(state)




        asd = 0


    def _update_actors(self):
        for log_actor_id in self.log_id_to_current_id_dict.keys():
            if [self.log_timestep] in self.actor_dics[log_actor_id]['time_step']:
                _actor_id = self.log_id_to_current_id_dict[log_actor_id]
                _actor = self.world.get_actor(_actor_id)
                index = self.find_corressponding_index(log_actor_id)

                log_location = self.actor_dics[log_actor_id]['loc'][index]
                log_rotation = self.actor_dics[log_actor_id]['rot'][index]
                log_speed = self.actor_dics[log_actor_id]['speed'][index]

                new_location = carla.Location(x=log_location[0], y=log_location[1], z=log_location[2])
                new_rotation = carla.Rotation(pitch=log_rotation[0], yaw=log_rotation[1], roll=log_rotation[2])
                new_speed = carla.Vector3D(x=log_speed[0], y=log_speed[1], z=log_speed[2])
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
                    print("_key:", _key)  # file[_key]
                    actor_id = _key.split('-')[1]
                    speed_key = '-'.join(_key.split('-')[:-1] + ['speed'])
                    control_key = '-'.join(_key.split('-')[:-1] + ['control'])
                    loc_key = '-'.join(_key.split('-')[:-1] + ['loc'])
                    rot_key = '-'.join(_key.split('-')[:-1] + ['rot'])
                    time_step = '-'.join(_key.split('-')[:-1] + ['time_step'])
                    self.actor_dics.update({actor_id: {'speed': np.array(file[speed_key]),
                                                       'actor_type': _key.split('-')[2],
                                                       'control': np.array(file[control_key]),
                                                       'loc': np.array(file[loc_key]), 'rot': np.array(file[rot_key]),
                                                       'time_step': np.array(file[time_step])}})
                    self.created_actor_dics.update({actor_id: {'actor_type': _key.split('-')[2],
                                                               'loc': np.array(file['new_' + loc_key]),
                                                               'rot': np.array(file['new_' + rot_key])}})
                    asd = 0
                elif 'traffic_light' in _key.split('-')[2].split('.') and actor_id not in self.actor_dics.keys():
                    print("_key:", _key)  # file[_key]
                    actor_id = _key.split('-')[1]
                    control_key = '-'.join(_key.split('-')[:-1] + ['state'])
                    loc_key = '-'.join(_key.split('-')[:-1] + ['loc'])
                    rot_key = '-'.join(_key.split('-')[:-1] + ['rot'])
                    time_step = '-'.join(_key.split('-')[:-1] + ['time_step'])
                    self.actor_dics.update({actor_id: {'actor_type': _key.split('-')[2],
                                                       'control': np.array(file[control_key]),
                                                       'loc': np.array(file[loc_key]), 'rot': np.array(file[rot_key]),
                                                       'time_step': np.array(file[time_step])}})

                asd = 0
        self.create_loged_actor_dic = {}
        self.end_time_loged_actor_dic = {}
        for index, key in enumerate(self.actor_dics.keys()):
            if self.actor_dics[key]['time_step'][0][0] not in self.create_loged_actor_dic.keys():
                self.create_loged_actor_dic.update({self.actor_dics[key]['time_step'][0][0]: [key]})
            else:
                self.create_loged_actor_dic[self.actor_dics[key]['time_step'][0][0]].append(key)

            if self.actor_dics[key]['time_step'][-1][0] not in self.end_time_loged_actor_dic.keys():
                self.end_time_loged_actor_dic.update({self.actor_dics[key]['time_step'][-1][0]: [key]})
            else:
                self.end_time_loged_actor_dic[self.actor_dics[key]['time_step'][-1][0]].append(key)

            asd = 0
        self.created_time_list = np.sort(list(self.create_loged_actor_dic.keys()))
        self.end_time_list = np.sort(list(self.end_time_loged_actor_dic.keys()))
        asd = 0

    def __call__(self):
        pass

    def destroy_actors(self, actor_list, ego_vehicle_id):
        for actor in actor_list:
            if actor.id != ego_vehicle_id:
                while actor.is_alive:
                    actor.destroy()
                    self.world.tick()
                    time.sleep(1)
                asd = 0

    def create_vehicle(self, ego_vehicle_id):

        if self.log_timestep == self.created_time_list[0]:
            # self.created_time_list.pop(0)
            self.created_time_list = np.delete(self.created_time_list, 0)
            if self.first_creation:
                self.destroy_actors(self.world.get_actors().filter('*vehicle*'), ego_vehicle_id)
                self.first_creation = False

            vehicle_id_list = self.create_loged_actor_dic[self.log_timestep]
            for vec in vehicle_id_list:
                # self.created_actor_dics[vec]['actor_type']
                vec_in_sim = self.create_in_simulation(vec, self.created_actor_dics[vec]['actor_type'],
                                                       self.created_actor_dics[vec]['loc'][0],
                                                       self.created_actor_dics[vec]['rot'][0])

                asd = 0

    def destroy_vehicle(self, ego_vehicle_id):
        if self.log_timestep == self.end_time_list[0]:
            self.end_time_list = np.delete(self.end_time_list, 0)
            vehicle_id_list = self.end_time_loged_actor_dic[self.log_timestep]
            for vec in vehicle_id_list:
                sim_vec_id = self.log_id_to_current_id_dict[vec]
                if ego_vehicle_id == sim_vec_id:
                    continue
                _actor = self.world.get_actor(sim_vec_id)
                while _actor.is_alive:
                    _actor.destroy()
                    self.world.tick()
                    time.sleep(1)

    def create_in_simulation(self, vec, type, location, rotaion):
        # Choose a vehicle blueprint
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(type)

        # Specify the spawn point

        # Spawn the vehicle
        vehicle = self.spawn_random_vehicle(vehicle_bp)

        new_transform = carla.Transform(carla.Location(x=location[0], y=location[1], z=location[2]),
                                        carla.Rotation(pitch=rotaion[0], yaw=rotaion[1], roll=rotaion[2]))
        vehicle.set_transform(new_transform)
        self.world.tick()
        time.sleep(1)

        # assert new_transform not in self.created_loc_list
        self.created_loc_list.append(new_transform)
        self.log_id_to_current_id_dict.update({vec: vehicle.id})
        return vehicle

    def spawn_random_vehicle(self, vehicle_bp):
        # Connect to the CARLA simulator running on localhost and the default port.
        # Get the world and its blueprint library.
        world = self.world
        blueprint_library = world.get_blueprint_library()

        # Choose a random vehicle blueprint.
        bp = vehicle_bp

        # Get a list of all possible spawn points and choose one at random.
        spawn_points = world.get_map().get_spawn_points()
        random_spawn_point = random.choice(spawn_points)

        # Spawn the vehicle.
        vehicle = None
        while vehicle == None:
            random_spawn_point = random.choice(spawn_points)
            vehicle = world.try_spawn_actor(bp, random_spawn_point)
            asd = 0
        return vehicle
