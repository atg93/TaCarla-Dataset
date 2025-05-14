import random

from leaderboard.autoagents.traditional_agents_files.carla_gym.core.task_actor.common.task_vehicle import TaskVehicle
import numpy as np
from importlib import import_module

import carla

PENALTY_COLLISION_PEDESTRIAN = 0.50
PENALTY_COLLISION_VEHICLE = 0.60
PENALTY_COLLISION_STATIC = 0.65
PENALTY_TRAFFIC_LIGHT = 0.70
PENALTY_STOP = 0.80


class EgoVehicleHandler(object):
    def __init__(self, client):
        self.ego_vehicles = {}
        self.info_buffers = {}
        self.reward_buffers = {}
        self.reward_handlers = {}
        self.terminal_handlers = {}

        self._world = client.get_world()
        self._map = self._world.get_map()
        self._spawn_transforms = self._get_spawn_points(self._map)
        self.traffic_light_disable = False

    def get_vehicle_and_world(self):
            return self.ego_vehicles['hero'].get_vehicle_and_world()

    def reset(self, task_config):
        actor_config = task_config['actors']
        route_config = task_config['routes']
        endless_config = task_config.get('endless')

        ev_spawn_locations = []
        #print("actor_config:",actor_config)
        for ev_id in actor_config:
            bp_filter = 'vehicle.lincoln.mkz_2020' #tugrul actor_config[ev_id]['model']
            blueprint = np.random.choice(self._world.get_blueprint_library().filter(bp_filter))
            blueprint.set_attribute('role_name', ev_id)

            if isinstance(route_config,dict) and 'hero' in route_config.keys():
                route = route_config[ev_id]
            else:
                route = route_config

            if len(route) == 0:
                spawn_transform = np.random.choice([x[1] for x in self._spawn_transforms])
            else:
                spawn_transform = route_config[ev_id][0]
            #Town3 stop signs
            #spawn_transform = carla.Transform(carla.Location(x=-74.700905, y=-107.961395, z=-0.343439), carla.Rotation(pitch=1.349543, yaw=-89.880669, roll=0.006048))
            #spawn_transform = self.change_initial_lane(spawn_transform)

            wp = self._map.get_waypoint(spawn_transform.location)

            spawn_transform.location.z = wp.transform.location.z + 1.321
            #spawn_transform.location, spawn_transform.rotation
            #spawn_transform = carla.Transform(carla.Location(x=-78.070801, y=-84.371796, z=-0.821804),
            #          carla.Rotation(pitch=0.715708, yaw=-90.026512, roll=0.002134))
            carla_vehicle = self._world.try_spawn_actor(blueprint, spawn_transform)
            self._world.tick()

            if endless_config is None:
                endless = False
            else:
                endless = endless_config[ev_id]

            target_transforms = route_config[ev_id][1:]
            #assert carla_vehicle == None
            #try:
            #    target_transforms = route_config[ev_id][1:]
            #except:
            #    target_transforms = [self._spawn_transforms[-10][1]]#tugrul

            self.ego_vehicles[ev_id] = TaskVehicle(carla_vehicle, target_transforms, self._spawn_transforms, endless)

            self.reward_handlers[ev_id] = self._build_instance(
                self._reward_configs[ev_id], self.ego_vehicles[ev_id])
            self.terminal_handlers[ev_id] = self._build_instance(
                self._terminal_configs[ev_id], self.ego_vehicles[ev_id])

            self.reward_buffers[ev_id] = []
            self.info_buffers[ev_id] = {
                'collisions_layout': [],
                'collisions_vehicle': [],
                'collisions_pedestrian': [],
                'collisions_others': [],
                'red_light': [],
                'encounter_light': [],
                'stop_infraction': [],
                'encounter_stop': [],
                'route_dev': [],
                'vehicle_blocked': [],
                'outside_lane': [],
                'wrong_lane': []
            }

            ev_spawn_locations.append(carla_vehicle.get_location())
        return ev_spawn_locations

    @staticmethod
    def _build_instance(config, ego_vehicle):
        module_str, class_str = config['entry_point'].split(':')
        _Class = getattr(import_module('carla_gym.core.task_actor.ego_vehicle.'+module_str), class_str)
        return _Class(ego_vehicle, **config.get('kwargs', {}))

    def apply_control(self, control_dict):
        for ev_id, control in control_dict.items():
            self.ego_vehicles[ev_id].vehicle.apply_control(control)

    def get_reward_value(self,timestamp,ev_id='hero',intersection_label=False):
        done, timeout, terminal_reward, terminal_debug = self.terminal_handlers[ev_id].get(timestamp)
        reward, reward_debug = self.reward_handlers[ev_id].get(terminal_reward, intersection_label)
        return reward, reward_debug

    def set_traffic_light_disable(self, traffic_light_disable):
        self.traffic_light_disable = traffic_light_disable
        for ev_id, ev in self.ego_vehicles.items():
            self.terminal_handlers[ev_id].set_traffic_light_disable(traffic_light_disable)

    def tick(self, timestamp):
        reward_dict, done_dict, info_dict = {}, {}, {}

        for ev_id, ev in self.ego_vehicles.items():
            info_criteria = ev.tick(timestamp)
            info = info_criteria.copy()
            done, timeout, terminal_reward, terminal_debug = self.terminal_handlers[ev_id].get(timestamp)
            reward, reward_debug = self.reward_handlers[ev_id].get(terminal_reward)

            reward_dict[ev_id] = reward
            done_dict[ev_id] = done
            info_dict[ev_id] = info
            info_dict[ev_id]['timeout'] = timeout
            info_dict[ev_id]['reward_debug'] = reward_debug
            info_dict[ev_id]['terminal_debug'] = terminal_debug

            # accumulate into buffers
            self.reward_buffers[ev_id].append(reward)

            if info['collision']:
                if info['collision']['collision_type'] == 0:
                    self.info_buffers[ev_id]['collisions_layout'].append(info['collision'])
                elif info['collision']['collision_type'] == 1:
                    self.info_buffers[ev_id]['collisions_vehicle'].append(info['collision'])
                elif info['collision']['collision_type'] == 2:
                    self.info_buffers[ev_id]['collisions_pedestrian'].append(info['collision'])
                else:
                    self.info_buffers[ev_id]['collisions_others'].append(info['collision'])
            if info['run_red_light']:
                self.info_buffers[ev_id]['red_light'].append(info['run_red_light'])
            if info['encounter_light']:
                self.info_buffers[ev_id]['encounter_light'].append(info['encounter_light'])
            if info['run_stop_sign']:
                if info['run_stop_sign']['event'] == 'encounter':
                    self.info_buffers[ev_id]['encounter_stop'].append(info['run_stop_sign'])
                elif info['run_stop_sign']['event'] == 'run':
                    self.info_buffers[ev_id]['stop_infraction'].append(info['run_stop_sign'])
            if info['route_deviation']:
                self.info_buffers[ev_id]['route_dev'].append(info['route_deviation'])
            if info['blocked']:
                self.info_buffers[ev_id]['vehicle_blocked'].append(info['blocked'])
            if info['outside_route_lane']:
                if info['outside_route_lane']['outside_lane']:
                    self.info_buffers[ev_id]['outside_lane'].append(info['outside_route_lane'])
                if info['outside_route_lane']['wrong_lane']:
                    self.info_buffers[ev_id]['wrong_lane'].append(info['outside_route_lane'])
            # save episode summary
            if done:
                info_dict[ev_id]['episode_event'] = self.info_buffers[ev_id]
                info_dict[ev_id]['episode_event']['timeout'] = info['timeout']
                info_dict[ev_id]['episode_event']['route_completion'] = info['route_completion']

                total_length = float(info['route_completion']['route_length_in_m']) / 1000
                completed_length = float(info['route_completion']['route_completed_in_m']) / 1000
                total_length = max(total_length, 0.001)
                completed_length = max(completed_length, 0.001)

                outside_lane_length = np.sum([x['distance_traveled']
                                              for x in self.info_buffers[ev_id]['outside_lane']]) / 1000
                wrong_lane_length = np.sum([x['distance_traveled']
                                            for x in self.info_buffers[ev_id]['wrong_lane']]) / 1000

                if ev._endless:
                    score_route = completed_length
                else:
                    if info['route_completion']['is_route_completed']:
                        score_route = 1.0
                    else:
                        score_route = completed_length / total_length

                n_collisions_layout = int(len(self.info_buffers[ev_id]['collisions_layout']))
                n_collisions_vehicle = int(len(self.info_buffers[ev_id]['collisions_vehicle']))
                n_collisions_pedestrian = int(len(self.info_buffers[ev_id]['collisions_pedestrian']))
                n_collisions_others = int(len(self.info_buffers[ev_id]['collisions_others']))
                n_red_light = int(len(self.info_buffers[ev_id]['red_light']))
                n_encounter_light = int(len(self.info_buffers[ev_id]['encounter_light']))
                n_stop_infraction = int(len(self.info_buffers[ev_id]['stop_infraction']))
                n_encounter_stop = int(len(self.info_buffers[ev_id]['encounter_stop']))
                n_collisions = n_collisions_layout + n_collisions_vehicle + n_collisions_pedestrian + n_collisions_others

                if self.traffic_light_disable:
                    n_red_light = 0
                    n_encounter_light = 0
                    n_stop_infraction = 0

                score_penalty = 1.0 * (1 - (outside_lane_length+wrong_lane_length)/completed_length) \
                    * (PENALTY_COLLISION_STATIC ** n_collisions_layout) \
                    * (PENALTY_COLLISION_VEHICLE ** n_collisions_vehicle) \
                    * (PENALTY_COLLISION_PEDESTRIAN ** n_collisions_pedestrian) \
                    * (PENALTY_TRAFFIC_LIGHT ** n_red_light) \
                    * (PENALTY_STOP ** n_stop_infraction) \

                if info['route_completion']['is_route_completed'] and n_collisions == 0:
                    is_route_completed_nocrash = 1.0
                else:
                    is_route_completed_nocrash = 0.0

                info_dict[ev_id]['episode_stat'] = {
                    'score_route': score_route,
                    'score_penalty': score_penalty,
                    'score_composed': max(score_route*score_penalty, 0.0),
                    'length': len(self.reward_buffers[ev_id]),
                    'reward': np.sum(self.reward_buffers[ev_id]),
                    'timeout': float(info['timeout']),
                    'is_route_completed': float(info['route_completion']['is_route_completed']),
                    'is_route_completed_nocrash': is_route_completed_nocrash,
                    'route_completed_in_km': completed_length,
                    'route_length_in_km': total_length,
                    'percentage_outside_lane': outside_lane_length / completed_length,
                    'percentage_wrong_lane': wrong_lane_length / completed_length,
                    'collisions_layout': n_collisions_layout / completed_length,
                    'collisions_vehicle': n_collisions_vehicle / completed_length,
                    'collisions_pedestrian': n_collisions_pedestrian / completed_length,
                    'collisions_others': n_collisions_others / completed_length,
                    'red_light': n_red_light / completed_length,
                    'light_passed': n_encounter_light-n_red_light,
                    'encounter_light': n_encounter_light,
                    'stop_infraction': n_stop_infraction / completed_length,
                    'stop_passed': n_encounter_stop-n_stop_infraction,
                    'encounter_stop': n_encounter_stop,
                    'route_dev': len(self.info_buffers[ev_id]['route_dev']) / completed_length,
                    'vehicle_blocked': len(self.info_buffers[ev_id]['vehicle_blocked']) / completed_length
                }

        done_dict['__all__'] = all(done for obs_id, done in done_dict.items())
        return reward_dict, done_dict, info_dict

    def clean(self):
        for ev_id, ev in self.ego_vehicles.items():
            ev.clean()
        self.ego_vehicles = {}
        self.reward_handlers = {}
        self.terminal_handlers = {}
        self.info_buffers = {}
        self.reward_buffers = {}

    @staticmethod
    def _get_spawn_points(c_map):
        all_spawn_points = c_map.get_spawn_points()

        spawn_transforms = []
        for trans in all_spawn_points:
            wp = c_map.get_waypoint(trans.location)

            if wp.is_junction:
                wp_prev = wp
                # wp_next = wp
                while wp_prev.is_junction:
                    wp_prev = wp_prev.previous(1.0)[0]
                spawn_transforms.append([wp_prev.road_id, wp_prev.transform])
                if c_map.name == 'Town03' and (wp_prev.road_id == 44):
                    for _ in range(100):
                        spawn_transforms.append([wp_prev.road_id, wp_prev.transform])
                # while wp_next.is_junction:
                #     wp_next = wp_next.next(1.0)[0]

                # spawn_transforms.append([wp_next.road_id, wp_next.transform])
                # if c_map.name == 'Town03' and (wp_next.road_id == 44 or wp_next.road_id == 58):
                #     for _ in range(100):
                #         spawn_transforms.append([wp_next.road_id, wp_next.transform])

            else:
                spawn_transforms.append([wp.road_id, wp.transform])
                if c_map.name == 'Town03' and (wp.road_id == 44):
                    for _ in range(100):
                        spawn_transforms.append([wp.road_id, wp.transform])

        return spawn_transforms

    def get_random_lane(self,given_wp):
        #check all lanes which are left side.
        current_lane_number = 0
        random_lane = random.randint(1,2)
        while given_wp != None and given_wp.lane_type == carla.LaneType.Driving:
            if random_lane == current_lane_number:
                break
            current_lane_number += 1
            given_wp = given_wp.get_left_lane()
        return given_wp

    def change_initial_lane(self,spawn_transform):
        wp = self._map.get_waypoint(spawn_transform.location)
        wp_avaliable = self.get_random_lane(wp)

        new_spawn_transform = carla.Transform(carla.Location(x=wp_avaliable.transform.location.x,
                                                             y=wp_avaliable.transform.location.y,
                                                             z=wp_avaliable.transform.location.z),spawn_transform.rotation)

        return new_spawn_transform





