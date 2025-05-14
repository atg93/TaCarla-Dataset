import copy
import math
import numpy as np
import cv2
from leaderboard.autoagents.traditional_agents_files.utils.process_radar import Process_Radar
import carla
from leaderboard.utils.route_manipulation import _location_to_gps

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from agents.navigation.global_route_planner import GlobalRoutePlanner

class Decision_of_Overtaking:
    def __init__(self):
        self.process_radar = Process_Radar()
        self.overtaking_count = 0
        overtaking_decision = False
        self.overtaking_threshold = 5
        hop_resolution = 1
        self.grp = GlobalRoutePlanner(CarlaDataProvider.get_map(), hop_resolution)
        # Obtain route plan
        self.lat_ref, self.lon_ref = 0, 0

    def interpolate_trajectory(self, waypoints_trajectory, hop_resolution=1.0):
        """
        Given some raw keypoints interpolate a full dense trajectory to be used by the user.
        returns the full interpolated route both in GPS coordinates and also in its original form.

        Args:
            - waypoints_trajectory: the current coarse trajectory
            - hop_resolution: distance between the trajectory's waypoints
        """
        route = []
        gps_route = []
        route_loc_list = []
        wpl_list = []

        for i in range(len(waypoints_trajectory) - 1):
            connection = 'fake_connection'
            waypoint = waypoints_trajectory[i]
            gps_coord = _location_to_gps(self.lat_ref, self.lon_ref, waypoint[0])
            gps_route.append((gps_coord, connection))

        return gps_route

    def __call__(self, input_data, fake_tl_masks, ev_mask, original_route_mask, wp_list, global_plan_gps, world_coordinate, high_level_action):
        #print("np.sum(ev_mask * original_route_mask):",np.sum(ev_mask * original_route_mask))
        is_there_light = np.sum(fake_tl_masks * original_route_mask).astype(np.int)

        self.original_wp_list = wp_list#copy.deepcopy(wp_list)
        new_wp_list, new_global_plan_gps, new_world_coordinate = wp_list, global_plan_gps, world_coordinate
        close_points_count,  carla_loc, img, mean_alt, mean_vel, is_there_obstacle = self.process_radar.show_radar_output(input_data['front_radar'],compass = input_data['imu'][1][-1])

        new_wp_list, new_global_plan_gps, new_world_coordinate = self.get_new_waypoints(wp_list, high_level_action)

        return img, close_points_count, mean_alt, mean_vel, is_there_obstacle, new_wp_list, new_global_plan_gps, new_world_coordinate

    def get_new_waypoints(self, wp_list, high_level_action):
        new_wp_list = []
        new_global_plan_gps = []
        new_world_coordinate = []
        for index, wp in enumerate(wp_list):
            addition = wp[1]
            waypoint = wp[0]
            if index < 15:
                if waypoint.is_intersection == False and waypoint.get_left_lane().lane_type==carla.LaneType.Driving and high_level_action == 'left_lane':
                    left_waypoint = waypoint.get_left_lane()
                    waypoint = left_waypoint
                elif waypoint.is_intersection == False and waypoint.get_right_lane().lane_type==carla.LaneType.Driving and high_level_action == 'right lane':
                    right_waypoint = waypoint.get_right_lane()
                    waypoint = right_waypoint
                new_wp_list.append((waypoint,addition))
                element = _location_to_gps(0, 0, waypoint.transform.location)
                new_global_plan_gps.append(element)
                new_world_coordinate.append((waypoint.transform.location,addition))
            else:
                new_wp_list.append((waypoint,addition))
                element = _location_to_gps(0,0,waypoint.transform.location)
                new_global_plan_gps.append(element)
                new_world_coordinate.append((waypoint.transform.location,addition))

        new_global_plan_gps = self.interpolate_trajectory(new_world_coordinate)


        return new_wp_list, new_global_plan_gps, new_world_coordinate


    def get_original_wp_list(self):
        return self.original_wp_list