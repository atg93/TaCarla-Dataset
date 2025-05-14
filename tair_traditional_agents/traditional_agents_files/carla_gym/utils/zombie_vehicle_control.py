import random
import numpy as np
import carla
import math

from carla_gym.core.task_actor.common.navigation.global_route_planner import GlobalRoutePlanner
from agents_roach.expert.utils.local_planner import LocalPlanner
import carla_gym.utils.transforms as trans_utils

class Zombie_Vehicle_Control():
    def __init__(self, world, zv_handler):
        self._world = world
        self._zv_handler = zv_handler
        self.zombie_vehicle_ids = []
        self._hack_throttle = False
        target_speed = 6.0
        resolution = 1.0
        longitudinal_pid_params = [0.5, 0.025, 0.1]
        lateral_pid_params = [0.75, 0.05, 0.0]
        threshold_before = 7.5
        threshold_after = 5.0
        self._local_planner = LocalPlanner(resolution, target_speed, longitudinal_pid_params,
                                           lateral_pid_params, threshold_before, threshold_after)

    def reset(self,map):
        self._map = map
        self.zombie_vehicle_ids = []
        self.lane_change_route_planner = GlobalRoutePlanner(self._map, resolution=1.0)
        self._local_planner.reset()
        self.last_wp = None
        self.reach_intersection = False

    def current_to_intersection(self, list_of_wp, actor):
        current_actor_location = actor.get_location()
        self.first_location = list_of_wp[0][0]
        current_way_point = self._map.get_waypoint(current_actor_location, project_to_road=True,
                                                   lane_type=(carla.LaneType.Driving))
        index = self.lane_change_route_planner._find_closest_in_list(current_way_point, np.array(list_of_wp)[:,0])

        self.intersection_wp = list_of_wp[index][0]
        self.route_intersection = self.lane_change_route_planner.trace_route(origin=current_actor_location,
                                                   destination=self.intersection_wp.transform.location)


    def new_wp_planner(self, actor, other_tl_location):
        current_actor_location = actor.get_location()

        self.last_wp_transform_location = other_tl_location
        if self.last_wp == None:
            new_route = self.lane_change_route_planner.trace_route(origin=self.intersection_wp.transform.location, destination=other_tl_location)
            new_route_1 = self.lane_change_route_planner.trace_route(origin=other_tl_location, destination=self.first_location.transform.location)
            self.new_route = self.route_intersection + new_route + new_route_1
            self.last_wp = self.new_route[-1]

        #print("self.intersection_wp.transform.location.distance(current_actor_location):",self.intersection_wp.transform.location.distance(current_actor_location))

        route_plan_dic = self.get_route_plan_dic(self.new_route, actor)

        return route_plan_dic


    def __call__(self, obs_dict, corresponding_tl_list, total_tl_list, list_of_wp):
        is_there_tl = obs_dict['hero']['birdview']['is_there_tl']
        is_there_vec = obs_dict['hero']['birdview']['is_there_vec']
        if not is_there_vec and is_there_tl and len(corresponding_tl_list) != 0 and len(total_tl_list) > 2:
            self._local_planner.reset()
            self.last_wp = None
            self.reach_intersection = False
            total_tl_list = list(total_tl_list)
            if corresponding_tl_list[0] in total_tl_list:
                total_tl_list.remove(corresponding_tl_list[0])

            random_tl_location = total_tl_list[random.randint(0, len(total_tl_list) - 1)]
            total_tl_list.remove(random_tl_location)
            self.other_tl_location = total_tl_list[random.randint(0, len(total_tl_list) - 1)]

            current_wp = self._map.get_waypoint(random_tl_location, project_to_road=True,
                                                lane_type=(carla.LaneType.Driving))
            new_wp = current_wp.next(distance=6)
            new_wp.append(current_wp)
            transform_list = [wp.transform for wp in new_wp]
            zombi_vec_info_dic = {}
            zombi_vec_info_dic['random_vec_transform'] = transform_list  # [new_wp[-1].transform]

            zombie_vehicle_ids = self._zv_handler.create_random_vehicle(zombi_vec_info_dic, autopilot=False)

            if len(self.zombie_vehicle_ids) != 0:
                destroyed_sucessfully = self.zombie_vehicle_ids[0].destroy()
                print("destroyed_sucessfully:",destroyed_sucessfully)

            if len(zombie_vehicle_ids) != 0:
                self.zombie_vehicle_ids = []
                self.create_actor_for_intersection = False
                self.zombie_vehicle_ids.append(zombie_vehicle_ids[0])
                self.current_to_intersection(list_of_wp=list_of_wp, actor=self.zombie_vehicle_ids[0])

        elif len(self.zombie_vehicle_ids) != 0:
            self.route_planer = self.new_wp_planner(actor=self.zombie_vehicle_ids[0],
                                                    other_tl_location=self.other_tl_location)
            speed_xy = math.sqrt(self.zombie_vehicle_ids[0].get_velocity().x**2 + self.zombie_vehicle_ids[0].get_velocity().y**2)
            control = self.run_pid_agent(self.route_planer, speed_xy)
            self.zombie_vehicle_ids[0].apply_control(control)


        if not is_there_vec:
            self.create_actor_for_intersection = True
            self.check_rotation = True

    def run_pid_agent(self,route_plan, speed_xy):
        throttle, steer, brake = self._local_planner.run_step(
            route_plan, speed_xy)
        if self._hack_throttle:
            throttle *= max((1.0 - abs(steer)), 0.25)

        control = carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
        return control

    def get_route_plan_dic(self, route_plan, actor):
        ev_transform = actor.get_transform()

        route_length = len(route_plan)
        location_list = []
        command_list = []
        road_id = []
        lane_id = []
        is_junction = []
        for i in range(route_length):
            if i < route_length:
                waypoint, road_option = route_plan[i]
            else:
                waypoint, road_option = route_plan[-1]

            wp_location_world_coord = waypoint.transform.location
            wp_location_actor_coord = trans_utils.loc_global_to_ref(wp_location_world_coord, ev_transform)
            location_list.append([wp_location_actor_coord.x, wp_location_actor_coord.y])
            command_list.append(road_option.value)
            road_id.append(waypoint.road_id)
            lane_id.append(waypoint.lane_id)
            is_junction.append(waypoint.is_junction)

        obs_dict = {
            'location': np.array(location_list, dtype=np.float32),
            'command': np.array(command_list, dtype=np.int8),
            'road_id': np.array(road_id, dtype=np.int8),
            'lane_id': np.array(lane_id, dtype=np.int8),
            'is_junction': np.array(is_junction, dtype=np.int8)
        }

        return obs_dict

    def new_wp_planner_yedek(self, list_of_wp, actor, other_tl_location):
        ego_rotation = list_of_wp[0][0].transform.rotation
        current_actor_location = actor.get_location()
        intersection_wp = list_of_wp[40][0]

        current_way_point = self._map.get_waypoint(current_actor_location, project_to_road=True,
                               lane_type=(carla.LaneType.Driving))

        last_way_point = self._map.get_waypoint(other_tl_location, project_to_road=True,
                               lane_type=(carla.LaneType.Driving))#.get_left_lane()

        distance = intersection_wp.transform.location.distance(current_actor_location)

        index = self.lane_change_route_planner._find_closest_in_list(current_way_point, np.array(list_of_wp)[:,0])
        """if self.last_wp == None:
            self.last_wp = list_of_wp[index][0]
            self.last_wp_transform_location = other_tl_location #self.last_wp.transform.location #current_way_point.next(distance)[-1] # #
        elif self.last_wp.transform.location.distance(current_actor_location) < 1:
            current_way_point = self._map.get_waypoint(current_actor_location, project_to_road=True,
                                                       lane_type=(carla.LaneType.Driving))
            self.last_wp = current_way_point.next(distance)[0]
            self.last_wp_transform_location = current_way_point.transform.location"""

        self.last_wp_transform_location = other_tl_location
        #check_rotation = ego_rotation == current_way_point.next(distance)[0].transform.rotation

        new_route = self.lane_change_route_planner.trace_route(origin=current_actor_location, destination=self.last_wp_transform_location)

        route_plan_dic = self.get_route_plan_dic(new_route, actor)

        return route_plan_dic


