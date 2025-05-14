import os
import h5py
import numpy as np
import carla
import copy


class Save_Il_Data():
    def __init__(self, current_path):
        self.current_scenario_info = None
        self.episode_number = 0
        self.dataset_folder_name = '/workspace/tg22/route_dataset_log_correction/' + current_path.split('/')[1]
        os.mkdir(self.dataset_folder_name)
        #self.read_control()
        self.new_actor_id = []
        self.less_new_actor_id = []

    def save_world(self, world):
        self.world = world

    def update_current_scenario_info(self, current_scenario_name, route_path,expert, file_name):
        self.current_scenario_name = current_scenario_name
        self.route_path = route_path
        self.expert = expert
        self.file_name = file_name
        self.reset_episode()
        self.episode_number += 1

    def save_control(self, control, ego_loc, ego_rot, ego_speed, button_press, ego_id):
        control_array = [control.throttle, control.steer, control.brake]
        ego_loc_array = [ego_loc.x, ego_loc.y, ego_loc.z]
        ego_rot_array = [ego_rot.pitch, ego_rot.yaw, ego_rot.roll]
        ego_speed_array = [ego_speed.x, ego_speed.y, ego_speed.z]
        vlen_str_dtype = h5py.special_dtype(vlen=str)
        if self.counter == 0:
            self.h5_file.create_dataset("control", data=np.expand_dims(control_array, axis=0),
                                        chunks=True, maxshape=(None,3))
            self.h5_file.create_dataset("ego_loc", data=np.expand_dims(ego_loc_array, axis=0),
                                        chunks=True, maxshape=(None, 3))
            self.h5_file.create_dataset("ego_rot", data=np.expand_dims(ego_rot_array, axis=0),
                                        chunks=True, maxshape=(None, 3))
            self.h5_file.create_dataset("ego_speed", data=np.expand_dims(ego_speed_array, axis=0),
                                        chunks=True, maxshape=(None, 3))

            vlen_str_dtype = h5py.special_dtype(vlen=str)
            # Create the dataset using this special dtype
            self.h5_file.create_dataset("current_scenario_name",
                                        data=np.expand_dims(np.array([self.current_scenario_name], dtype=object),
                                                            axis=0), dtype=vlen_str_dtype,
                                        chunks=True, maxshape=(None, 1,))

            self.h5_file.create_dataset("route_path",
                                        data=np.expand_dims(np.array([self.route_path], dtype=object),
                                                            axis=0), dtype=vlen_str_dtype,
                                        chunks=True, maxshape=(None, 1,))

            self.h5_file.create_dataset("expert",
                                        data=np.expand_dims(np.array([self.expert], dtype=object),
                                                            axis=0), dtype=vlen_str_dtype,
                                        chunks=True, maxshape=(None, 1,))

            self.h5_file.create_dataset("file_name",
                                        data=np.expand_dims(np.array([self.file_name], dtype=object),
                                                            axis=0), dtype=vlen_str_dtype,
                                        chunks=True, maxshape=(None, 1,))

            self.h5_file.create_dataset("current_scenario_label", data=np.expand_dims(np.array([int(button_press)]), axis=0),
                                        chunks=True, maxshape=(None, 1))




        else:  # append next arrays to dataset
            self.h5_file["current_scenario_label"].resize((self.h5_file["current_scenario_label"].shape[0] + 1), axis=0)
            self.h5_file["current_scenario_label"][-1] = np.array([int(button_press)])
            self.h5_file["control"].resize((self.h5_file["control"].shape[0] + 1), axis=0)
            self.h5_file["control"][-1] = control_array
            self.h5_file["ego_loc"].resize((self.h5_file["ego_loc"].shape[0] + 1), axis=0)
            self.h5_file["ego_loc"][-1] = ego_loc_array
            self.h5_file["ego_rot"].resize((self.h5_file["ego_rot"].shape[0] + 1), axis=0)
            self.h5_file["ego_rot"][-1] = ego_rot_array
            self.h5_file["ego_speed"].resize((self.h5_file["ego_speed"].shape[0] + 1), axis=0)
            self.h5_file["ego_speed"][-1] = ego_speed_array
        self.check_other_actors(ego_loc, vlen_str_dtype,ego_id)
        self.counter += 1

    def read_control(self):
        self.control_data_index = 0
        filename = self.dataset_folder_name + '/episode_' + str('024').zfill(3) + '.h5'
        self.control_data = []
        with h5py.File(filename, 'r') as file:
            # Access a dataset within the file
            # Replace 'dataset_name' with the name of the dataset you want to access
            self.control_data.append(file['control'][:])

    def check_other_actors(self,ego_loc,vlen_str_dtype,ego_id):
        actor_list = self.world.get_actors()

        vehicle_actor_list = []
        for actor in actor_list:
            if 'vehicle' in actor.type_id.split('.'): # and 'traffic_light' not in actor.type_id.split('.'):
                vehicle_actor_list.append(actor)

            if 'traffic' in actor.type_id.split('.'):
                vehicle_actor_list.append(actor)

            if 'walker' in actor.type_id.split('.'):
                vehicle_actor_list.append(actor)

        # Print all actors
        for actor in vehicle_actor_list:

            if ego_id == actor.id:
                continue

            location1 = actor.get_transform().location
            distance = location1.distance(ego_loc)

            if actor.id not in self.new_actor_id:
                actor_loc = actor.get_transform().location
                actor_rot = actor.get_transform().rotation
                actor_loc_array = copy.deepcopy([actor_loc.x, actor_loc.y, actor_loc.z])
                actor_rot_array = copy.deepcopy([actor_rot.pitch, actor_rot.yaw, actor_rot.roll])
                
                self.h5_file.create_dataset('new_actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'loc',
                                            data=np.expand_dims(actor_loc_array, axis=0),
                                            chunks=True, maxshape=(None, 3))
                self.h5_file.create_dataset('new_actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'rot',
                                            data=np.expand_dims(actor_rot_array, axis=0),
                                            chunks=True, maxshape=(None, 3))
                self.new_actor_id.append(actor.id)

            if distance < 200 and 'sensor' not in actor.type_id.split('.'):

                #print(f'Actor ID: {actor.id}, Actor Type: {actor.type_id}')
                actor_loc = actor.get_transform().location
                actor_rot = actor.get_transform().rotation
                actor_loc_array = [actor_loc.x, actor_loc.y, actor_loc.z]
                actor_rot_array = [actor_rot.pitch, actor_rot.yaw, actor_rot.roll]
                if 'actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'loc' not in self.h5_file.keys():
                    self.h5_file.create_dataset('actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'loc',
                                                data=np.expand_dims(actor_loc_array, axis=0),
                                                chunks=True, maxshape=(None, 3))
                    self.h5_file.create_dataset('actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'rot',
                                                data=np.expand_dims(actor_rot_array, axis=0),
                                                chunks=True, maxshape=(None, 3))
                    self.h5_file.create_dataset('actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'time_step',
                                                data=np.expand_dims([self.counter], axis=0),
                                                chunks=True, maxshape=(None, 1))



                    if 'traffic_light' in actor.type_id.split('.'):#traffic.traffic_light
                        state = actor.get_state()
                        self.h5_file.create_dataset('actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'state',
                                                    data=np.expand_dims(np.array([str(state)], dtype=object),
                                                                        axis=0), dtype=vlen_str_dtype,
                                                    chunks=True, maxshape=(None, 1,))

                    if 'vehicle' in actor.type_id.split('.'):
                        actor_control = actor.get_control()
                        actor_speed = actor.get_velocity()
                        actor_speed_array = [actor_speed.x, actor_speed.y, actor_speed.z]
                        actor_control_array = np.array([actor_control.steer, actor_control.throttle, actor_control.brake])

                        self.h5_file.create_dataset('actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' +"speed", data=np.expand_dims(actor_speed_array, axis=0),
                                                    chunks=True, maxshape=(None, 3))
                        self.h5_file.create_dataset('actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' +"control", data=np.expand_dims(actor_control_array, axis=0),
                                                    chunks=True, maxshape=(None, 3))

                else:
                    self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'loc'].resize(
                        (self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'loc'].shape[0] + 1),
                        axis=0)
                    self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'loc'][-1] = actor_loc_array

                    self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'rot'].resize(
                        (self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'rot'].shape[0] + 1),
                        axis=0)
                    self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'rot'][-1] = actor_rot_array

                    self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'time_step'].resize(
                        (self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'time_step'].shape[0] + 1),
                        axis=0)
                    self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'time_step'][-1] = np.array([self.counter])

                    if 'traffic_light' in actor.type_id.split('.'):
                        state = actor.get_state()
                        self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'state'].resize(
                            (self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'state'].shape[0] + 1),
                            axis=0)
                        self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' + 'state'][-1] = str(state)

                    if 'vehicle' in actor.type_id.split('.'):
                        actor_speed = actor.get_velocity()
                        actor_speed_array = np.array([actor_speed.x, actor_speed.y, actor_speed.z])
                        self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' +"speed"].resize((self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' +"speed"].shape[0] + 1), axis=0)
                        self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' +"speed"][-1] = actor_speed_array

                        actor_control = actor.get_control()
                        actor_control_array = np.array([actor_control.steer, actor_control.throttle, actor_control.brake])
                        self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' +"control"].resize((self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' +"control"].shape[0] + 1), axis=0)
                        self.h5_file['actor' + '-' + str(actor.id) + '-' + actor.type_id + '-' +"control"][-1] = actor_control_array

        asd = 0






    def get_control(self):
        control = self.control_data[0][self.control_data_index]
        carla_control = carla.VehicleControl(
            throttle=control[0],
            steer=control[1],
            brake=control[2]
        )
        self.control_data_index += 1

        return carla_control


    def save_score(self,score_composed, score_route, score_penalty):
        print("score_composed: ", score_composed)
        self.h5_file.create_dataset("score_composed", data=np.expand_dims([score_composed], axis=0),
                                    chunks=True, maxshape=(None, 1))

        self.h5_file.create_dataset("score_route", data=np.expand_dims([score_route], axis=0),
                                    chunks=True, maxshape=(None, 1))

        self.h5_file.create_dataset("score_penalty", data=np.expand_dims([score_penalty], axis=0),
                                    chunks=True, maxshape=(None, 1))

    def save_agentblocked(self,agent_blocked_score):
        self.h5_file.create_dataset("agent_blocked_score",
                                    data=np.expand_dims(np.array([int(agent_blocked_score)]), axis=0),
                                    chunks=True, maxshape=(None, 1))



    def reset_episode(self):

        self.vehicle_dic_dataset = {}
        self.vehicle_index_dataset = 1

        self.counter = 0
        episode_number = self.episode_number
        file = self.dataset_folder_name + '/episode_' + str(episode_number).zfill(3) + '.h5'
        file_exist = True

        while file_exist:
            episode_number += 1
            file = self.dataset_folder_name + '/episode_' + str(episode_number).zfill(3) + '.h5'
            file_exist = os.path.exists(file)

        self.h5_file = h5py.File(file, 'w')


    def close_h5_file(self):
        self.h5_file.close()
