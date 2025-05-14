import cv2
import numpy as np
import carla
from gym import spaces
import cv2 as cv
from collections import deque
from pathlib import Path
import h5py

from leaderboard.autoagents.traditional_agents_files.carla_gym.core.obs_manager.obs_manager import ObsManagerBase
from leaderboard.autoagents.traditional_agents_files.carla_gym.utils.traffic_light import TrafficLightHandler
from leaderboard.autoagents.traditional_agents_files.perception.detection_utils.geometry import scale_and_zoom, get_pose_matrix, euler_to_quaternion
import torch

import os
import time
import math

COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_5 = (46, 52, 54)


def tint(color, factor):
    r, g, b = color
    r = int(r + (255-r) * factor)
    g = int(g + (255-g) * factor)
    b = int(b + (255-b) * factor)
    r = min(r, 255)
    g = min(g, 255)
    b = min(b, 255)
    return (r, g, b)


class ObsManager(ObsManagerBase):
    def __init__(self, obs_configs):
        self._width = int(obs_configs['width_in_pixels'])
        self._pixels_ev_to_bottom = obs_configs['pixels_ev_to_bottom']
        self._pixels_per_meter = obs_configs['pixels_per_meter']
        self._history_idx = obs_configs['history_idx']
        self._scale_bbox = obs_configs.get('scale_bbox', True)
        self._scale_mask_col = obs_configs.get('scale_mask_col', 1.1)

        self._history_queue = deque(maxlen=20)
        self.ego_queue = deque(maxlen=20)
        self.box_que = deque(maxlen=20)

        self._image_channels = 3
        self._masks_channels = 3 + 3*len(self._history_idx)
        self._parent_actor = None
        self._world = None

        self._map_dir = Path(__file__).resolve().parent / 'maps'

        super(ObsManager, self).__init__()

        self.compute_prediction_input = None  # Compute_Prediction_Input(self._width, self._pixels_per_meter,self._pixels_ev_to_bottom, self)


        self._ego_motion_queue = deque(maxlen=3)
        self.sensor = None
        self.sensor_interface = None
        self.control_with = 'plant'
        self.route_plan = None
        self.prev_ev_loc = None
        #self.bev_resolution =
        #self.view_array =

    def _define_obs_space(self):
        self.obs_space = spaces.Dict(
            {'rendered': spaces.Box(
                low=0, high=255, shape=(self._width, self._width, self._image_channels),
                dtype=np.uint8),
             'masks': spaces.Box(
                low=0, high=255, shape=(self._masks_channels, self._width, self._width),
                dtype=np.uint8)})

    def attach_ego_vehicle(self, vehicle):
        self.vehicle = vehicle
        self._parent_actor = vehicle
        self._world = vehicle.get_world()

        maps_h5_path = ''.join(np.array(list(str(self._map_dir)))[53:]) + '/' + (self._world.get_map().name + '.h5')[11:]
        path_exist = os.path.exists(maps_h5_path)
        if not path_exist:
            maps_h5_path = 'leaderboard' + '/' + maps_h5_path

        print("maps_h5_path:",maps_h5_path)
        assert os.path.exists(maps_h5_path)
        with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf:
            self._world_offset = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)
            self._pixels_per_meter = float(hf.attrs['pixels_per_meter'])
            assert np.isclose(self._pixels_per_meter, float(hf.attrs['pixels_per_meter']))
        self._distance_threshold = np.ceil(self._width / self._pixels_per_meter)
        return self._pixels_per_meter, self._world_offset

    @staticmethod
    def _get_stops(criteria_stop, veh_loc):
        stop_sign = criteria_stop._target_stop_sign
        stops = []
        data_stops = []
        bb_list = []
        if (stop_sign is not None) and (not criteria_stop._stop_completed):
            bb_loc = carla.Location(stop_sign.trigger_volume.location)
            bb_ext = carla.Vector3D(stop_sign.trigger_volume.extent)
            bb_ext.x = max(bb_ext.x, bb_ext.y)
            bb_ext.y = max(bb_ext.x, bb_ext.y)
            trans = stop_sign.get_transform()
            stops = [(carla.Transform(trans.location, trans.rotation), bb_loc, bb_ext)]

        if (stop_sign is not None):
            bb_loc = carla.Location(stop_sign.trigger_volume.location)
            bb_ext = carla.Vector3D(stop_sign.trigger_volume.extent)
            bb_ext.x = max(bb_ext.x, bb_ext.y)
            bb_ext.y = max(bb_ext.x, bb_ext.y)
            trans = stop_sign.get_transform()
            data_stops = [(carla.Transform(trans.location, trans.rotation), bb_loc, bb_ext)]
        return stops, data_stops
    def update_route_plant(self,ev_loc):
        if self.route_plan[0][0].transform.location.distance(ev_loc)<3:
            self.route_plan = self.route_plan[1:]
            print("len(self.route_plan):",len(self.route_plan))

    def get_ego(self, ev_transform):
        ego = {}
        ego['rotation'] = euler_to_quaternion(-ev_transform.rotation.roll, ev_transform.rotation.pitch,
                                              -ev_transform.rotation.yaw)
        ego['translation'] = ev_transform.location.x, -ev_transform.location.y, ev_transform.location.z
        return ego

    def get_plant_ego(self, ev_transform, prev_ev_transform):
        ego = {}
        ego['rotation'] = euler_to_quaternion(-ev_transform.rotation.roll, ev_transform.rotation.pitch,
                                              -ev_transform.rotation.yaw)
        ego['translation'] = ev_transform.location.x, -ev_transform.location.y, ev_transform.location.z
        dx, dy = ev_transform.location.x, -ev_transform.location.y
        ego_motion = dx, dy, 0, 0, yaw, 0
        return ego

    def get_observation(self, route_plan, plant_global_plan_gps, plant_global_plan_world_coord, detected_masks=None, detected_bbox=None, unwbbox=None):
        if self.route_plan == None:
            self.route_plan = route_plan
        else:
            route_plan = self.route_plan

        ev_transform = self.vehicle.get_transform()
        ego = self.get_ego(ev_transform)
        self.ego_queue.append(ego)
        self.box_que.append((detected_bbox, unwbbox))

        ev_loc = ev_transform.location
        ev_rot = ev_transform.rotation
        ev_bbox = self.vehicle.bounding_box
        snap_shot = self._world.get_snapshot()
        self.update_route_plant(ev_loc)
        if self.prev_ev_loc==None:
            self.prev_ev_loc = ev_loc
            self.prev_ev_transform = ev_transform

        def is_within_distance(w):
            c_distance = abs(ev_loc.x - w.location.x) < self._distance_threshold \
                and abs(ev_loc.y - w.location.y) < self._distance_threshold \
                and abs(ev_loc.z - w.location.z) < 8.0
            c_ev = abs(ev_loc.x - w.location.x) < 1.0 and abs(ev_loc.y - w.location.y) < 1.0
            return c_distance and (not c_ev)

        vehicle_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Car)
        walker_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Pedestrians)
        if self._scale_bbox:
            vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance, 1.0)
            walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance, 2.0)
        else:
            vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance)
            walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance)

        tl_green, green_list = TrafficLightHandler.get_stopline_vtx(ev_loc, 0)
        tl_yellow, yel_list = TrafficLightHandler.get_stopline_vtx(ev_loc, 1)
        tl_red, red_list = TrafficLightHandler.get_stopline_vtx(ev_loc, 2)
        stops, data_stops = [], [] #data_stops = self._get_stops(self._parent_actor.criteria_stop, ev_loc)
        self.tl_corner_bb_list = green_list + yel_list + red_list
        #self.stops = data_stops

        self._history_queue.append((vehicles, walkers, tl_green, tl_yellow, tl_red, stops, data_stops))

        M_warp = self._get_warp_transform(ev_loc, ev_rot)

        # objects with history
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks, data_stops_masks \
            = self._get_history_masks(M_warp)

        road_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        lane_mask_all = np.zeros([self._width, self._width], dtype=np.uint8)
        lane_mask_broken = np.zeros([self._width, self._width], dtype=np.uint8)

        # ev_mask
        ev_mask = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location, ev_bbox.extent)], M_warp)
        ev_mask_col = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location,
                                                       ev_bbox.extent*self._scale_mask_col)], M_warp)

        # route_mask
        route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                   for wp, _ in route_plan[0:80]])
        route_warped = cv.transform(route_in_pixel, M_warp)
        cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=16)
        route_mask = route_mask.astype(np.bool)

        tl_route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        tl_route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                   for wp, _ in route_plan[0:20]])
        tl_route_warped = cv.transform(tl_route_in_pixel, M_warp)
        cv.polylines(tl_route_mask, [np.round(tl_route_warped).astype(np.int32)], False, 1, thickness=1)
        self.tl_route_masks = tl_route_mask.astype(np.bool)

        stop_route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        tl_route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                      for wp, _ in route_plan[5:20]])
        tl_route_warped = cv.transform(tl_route_in_pixel, M_warp)
        cv.polylines(stop_route_mask, [np.round(tl_route_warped).astype(np.int32)], False, 1, thickness=1)
        self.stop_route_mask = stop_route_mask.astype(np.bool)


        planning_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        #route plannig
        control = None
        targetpoint_mask = None
        light_hazard = None
        attention_mask = np.zeros([self._width, self._width, 3], dtype=np.uint8)
        keep_vehicle_ids = []
        keep_vehicle_attn = []
        t1 = time.time()
        if type(self.sensor_interface) != type(None):
            pass

        #t2 = time.time()
        #print("t2-t1:", t2 - t1)
        # render
        image = np.zeros([self._width, self._width, 3], dtype=np.uint8)
        image[road_mask] = COLOR_ALUMINIUM_5
        image[route_mask] = COLOR_ALUMINIUM_3
        if type(targetpoint_mask) != type(None):
            image[targetpoint_mask] = COLOR_RED

        image[lane_mask_all] = COLOR_MAGENTA
        image[lane_mask_broken] = COLOR_MAGENTA_2

        h_len = len(self._history_idx)-1
        for i, mask in enumerate(stop_masks):
            image[mask] = tint(COLOR_YELLOW_2, (h_len-i)*0.2)
        for i, mask in enumerate(tl_green_masks):
            image[mask] = tint(COLOR_GREEN, (h_len-i)*0.2)
        for i, mask in enumerate(tl_yellow_masks):
            image[mask] = tint(COLOR_YELLOW, (h_len-i)*0.2)
        for i, mask in enumerate(tl_red_masks):
            image[mask] = tint(COLOR_RED, (h_len-i)*0.2)

        self.set_is_there_list_parameter(data_stops_masks, tl_green_masks, tl_yellow_masks, tl_red_masks)

        image = self.draw_attention_bb(image, keep_vehicle_ids, keep_vehicle_attn, M_warp, vehicle_masks)
        if isinstance(detected_masks,type(None)):
            for i, mask in enumerate(vehicle_masks):
                image[mask] = tint(COLOR_BLUE, (h_len-i)*0.2)
            for i, mask in enumerate(walker_masks):
                image[mask] = tint(COLOR_CYAN, (h_len-i)*0.2)
        elif not isinstance(detected_bbox,type(None)):
            i = 0
            detection_mask = np.zeros([self._width, self._width, 3], dtype=np.uint8)
            mask = self.convert_bbox2mask(detected_bbox)
            image[mask] = 255

        else:
            i = 0
            image[detected_masks[0]>0] = tint(COLOR_BLUE, (h_len - i) * 0.2)
            image[detected_masks[1]>0] = tint(COLOR_CYAN, (h_len - i) * 0.2)



        image[ev_mask] = COLOR_WHITE

        if self.control_with == 'plant':
            image[planning_mask] = COLOR_BLUE

        # image[obstacle_mask] = COLOR_BLUE

        # masks
        c_road = road_mask * 255
        c_route = route_mask * 255
        c_lane = lane_mask_all * 255
        c_lane[lane_mask_broken] = 120

        # masks with history
        c_tl_history = []
        for i in range(len(self._history_idx)):
            c_tl = np.zeros([self._width, self._width], dtype=np.uint8)
            c_tl[tl_green_masks[i]] = 80
            c_tl[tl_yellow_masks[i]] = 170
            c_tl[tl_red_masks[i]] = 255
            c_tl[stop_masks[i]] = 255
            c_tl_history.append(c_tl)

        c_vehicle_history = [m*255 for m in vehicle_masks]
        c_walker_history = [m*255 for m in walker_masks]

        masks = np.stack((c_road, c_route, c_lane, *c_vehicle_history, *c_walker_history, *c_tl_history), axis=2)
        masks = np.transpose(masks, [2, 0, 1])
        ev_loc_in_px = self._world_to_pixel(ev_loc)
        prev_ev_loc_in_px = self._world_to_pixel(self.prev_ev_loc)

        yaw = self.calculate_angle(ev_loc_in_px[0],ev_loc_in_px[1],prev_ev_loc_in_px[0],prev_ev_loc_in_px[1])#ev_transform.rotation.yaw
        plant_motion = [(ev_loc_in_px - prev_ev_loc_in_px)[0], (ev_loc_in_px - prev_ev_loc_in_px)[1], 0, 0, yaw, 0]#dx, dy, _, _, yaw, _
        ego_motion = self.get_ego_motion()
        #ego_motion = self.get_plant_ego(ev_transform)
        plant_motion = [(ev_loc_in_px - prev_ev_loc_in_px)[0], (ev_loc_in_px - prev_ev_loc_in_px)[1], 0, 0, ev_transform.rotation.yaw-self.prev_ev_transform.rotation.yaw, 0]#dx, dy, _, _, yaw, _
        #M = cv2.getAffineTransform(prev_ev_loc_in_px, ev_loc_in_px)
        obs_dict = {'rendered': image, 'masks': masks, 'plant_control': control,'light_hazard':light_hazard,'ego_motion':plant_motion}

        cv2.imwrite("image.png",image)

        #self._parent_actor.collision_px = np.any(ev_mask_col & walker_masks[-1])
        self.prev_ev_loc = ev_loc
        self.prev_ev_transform = ev_transform

        return obs_dict

    def calculate_angle(self, x1, y1, x2, y2, epsilon=1e-10):
        # Calculate the slope with epsilon to avoid division by zero
        slope = (y2 - y1) / ((x2 - x1) + epsilon)

        # Calculate the angle in radians
        angle_radians = math.atan(slope)

        # Convert the angle to degrees
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees

    def set_info(self,info):
        self.bev_resolution, self.view_array = info

    def get_ego_motion_0(self):
        ego_motions = torch.zeros((1, 3, 4, 4))
        ego_motions[0, :] = torch.eye(4)
        for i, idx in enumerate(self._history_idx):
            idx = max(idx, -1 * len(self.ego_queue))
            if i == 3:
                continue
            next_idx = max(self._history_idx[i + 1], -1 * len(self.ego_queue))
            ego_motions[0, i] = torch.from_numpy(self.get_relative_rt(self.ego_queue[idx], self.ego_queue[next_idx], self.view_array))
            self.box_que
            # ego_motions[0, i] = torch.from_numpy(np.linalg.inv(ego_queue[idx]) @ ego_queue[next_idx])

        return ego_motions

    def get_ego_motion(self):
        ego_motions = torch.zeros((1, 3, 4, 4))
        ego_motions[0, :] = torch.eye(4)
        for i, idx in enumerate(self._history_idx):
            idx = max(idx, -1 * len(self.ego_queue))
            if i == 3:
                continue
            next_idx = max(self._history_idx[i + 1], -1 * len(self.ego_queue))
            ego_motions[0, i] = torch.from_numpy(self.get_relative_rt(self.ego_queue[idx], self.ego_queue[next_idx], self.view_array))
            self.box_que
            # ego_motions[0, i] = torch.from_numpy(np.linalg.inv(ego_queue[idx]) @ ego_queue[next_idx])

        return ego_motions

    def get_relative_rt(self, previous_ego, ego, view):

        lidar_to_world_t0 = get_pose_matrix(previous_ego, use_flat=False)
        lidar_to_world_t1 = get_pose_matrix(ego, use_flat=False)
        future_egomotion = np.linalg.inv(lidar_to_world_t1) @ lidar_to_world_t0

        sh, sw, _ = 1 / self.bev_resolution
        view_rot_only = np.eye(4, dtype=np.float32)
        view_rot_only[0, 0:2] = view[0, 0, 0, 0, 0:2] / sw
        view_rot_only[1, 0:2] = view[0, 0, 0, 1, 0:2] / sh
        future_egomotion = view_rot_only @ future_egomotion @ np.linalg.inv(view_rot_only)
        future_egomotion[3, :3] = 0.0
        future_egomotion[3, 3] = 1.0

        return future_egomotion

    def _get_history_masks(self, M_warp):
        qsize = len(self._history_queue)
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks, data_stops_masks = [], [], [], [], [], [], []
        for idx in self._history_idx:
            idx = max(idx, -1 * qsize)

            vehicles, walkers, tl_green, tl_yellow, tl_red, stops, data_stops = self._history_queue[idx]

            vehicle_masks.append(self._get_mask_from_actor_list(vehicles, M_warp))
            walker_masks.append(self._get_mask_from_actor_list(walkers, M_warp))
            tl_green_masks.append(self._get_mask_from_stopline_vtx(tl_green, M_warp))
            tl_yellow_masks.append(self._get_mask_from_stopline_vtx(tl_yellow, M_warp))
            tl_red_masks.append(self._get_mask_from_stopline_vtx(tl_red, M_warp))
            stop_masks.append(self._get_mask_from_actor_list(stops, M_warp))


            data_stops_masks.append(self._get_mask_from_actor_list(data_stops, M_warp))

        return vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks, data_stops_masks

    def _get_mask_from_stopline_vtx(self, stopline_vtx, M_warp):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        for sp_locs in stopline_vtx:
            stopline_in_pixel = np.array([[self._world_to_pixel(x)] for x in sp_locs])
            stopline_warped = cv.transform(stopline_in_pixel, M_warp)
            cv.line(mask, tuple(stopline_warped[0, 0]), tuple(stopline_warped[1, 0]),
                    color=1, thickness=6)
        return mask.astype(np.bool)

    def _get_mask_from_actor_list(self, actor_list, M_warp):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        for actor_transform, bb_loc, bb_ext in actor_list:

            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=0),
                       carla.Location(x=bb_ext.x, y=bb_ext.y),
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]
            corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])
            corners_warped = cv.transform(corners_in_pixel, M_warp)

            cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
        return mask.astype(np.bool)

    @staticmethod
    def _get_surrounding_actors(bbox_list, criterium, scale=None):
        actors = []
        for bbox in bbox_list:
            is_within_distance = criterium(bbox)
            if is_within_distance:
                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bbox.extent)
                if scale is not None:
                    bb_ext = bb_ext * scale
                    bb_ext.x = max(bb_ext.x, 0.8)
                    bb_ext.y = max(bb_ext.y, 0.8)

                actors.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
        return actors

    def _get_warp_transform(self, ev_loc, ev_rot):
        ev_loc_in_px = self._world_to_pixel(ev_loc)
        yaw = np.deg2rad(ev_rot.yaw)

        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        right_vec = np.array([np.cos(yaw + 0.5*np.pi), np.sin(yaw + 0.5*np.pi)])

        bottom_left = ev_loc_in_px - self._pixels_ev_to_bottom * forward_vec - (0.5*self._width) * right_vec
        top_left = ev_loc_in_px + (self._width-self._pixels_ev_to_bottom) * forward_vec - (0.5*self._width) * right_vec
        top_right = ev_loc_in_px + (self._width-self._pixels_ev_to_bottom) * forward_vec + (0.5*self._width) * right_vec

        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.array([[0, self._width-1],
                            [0, 0],
                            [self._width-1, 0]], dtype=np.float32)
        return cv.getAffineTransform(src_pts, dst_pts)

    def _world_to_pixel(self, location, projective=False):
        """Converts the world coordinates to pixel coordinates"""
        x = self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self._pixels_per_meter * (location.y - self._world_offset[1])

        if projective:
            p = np.array([x, y, 1], dtype=np.float32)
        else:
            p = np.array([x, y], dtype=np.float32)
        return p

    def _world_to_pixel_width(self, width):
        """Converts the world units to pixel units"""
        return self._pixels_per_meter * width

    def set_is_there_list_parameter(self, stop_masks, tl_green_masks, tl_yellow_masks, tl_red_masks):
        element_1 = np.zeros([self._width, self._width], dtype=np.uint8)
        element_1[(stop_masks[0] + tl_green_masks[0] + tl_yellow_masks[0] + tl_red_masks[0])] = 255

        stop_sign = np.sum((stop_masks[-1] * self.stop_route_mask)).astype(np.bool)
        green_sign = np.sum((tl_green_masks[-1] * self.tl_route_masks)).astype(np.bool)
        yellow_sign = np.sum((tl_yellow_masks[-1] * self.tl_route_masks)).astype(np.bool)
        red_sign = np.sum((tl_red_masks[-1] * self.tl_route_masks)).astype(np.bool)
        #cv2.imwrite("stop_masks.png", stop_masks[-1].astype(np.uint8)*255)
        #cv2.imwrite("tl_route_masks.png", self.tl_route_masks.astype(np.uint8)*255)

        self.color = "unknown"
        if np.sum(stop_sign).astype(np.bool):
            self.color = "stop_sign"
        if np.sum(red_sign).astype(np.bool):
            self.color = "red"
        if np.sum(yellow_sign).astype(np.bool):
            self.color = "yellow"
        if np.sum(green_sign).astype(np.bool):
            self.color = "green"

    def get_is_there_list_parameter(self):
        stop_box = []
        """for actor in self._parent_actor.criteria_stop._list_stop_signs:
            bb = carla.BoundingBox(actor.get_transform().location, actor.bounding_box.extent)
            bb.rotation = actor.get_transform().rotation
            stop_box.append(bb)"""
        return self.color, stop_box, self.tl_corner_bb_list

    def set_sensor(self, sensors):
        # gps = np.array([sensors[0]['lat'], sensors[0]['lon']])#['compass']
        self.sensor = {'gps': np.array([sensors[0]['lat'], sensors[0]['lon']]), 'imu': sensors[1], 'sensor': sensors}


    def set_control_with(self,_state):
        self.control_with = _state

    def draw_attention_bb(self, attention_mask, keep_vehicle_ids, keep_vehicle_attn, M_warp, real_vehicle_masks, scale=1.5):
        actors = self._world.get_actors()
        all_vehicles = actors.filter('*vehicle*')
        all_walkers = actors.filter('*pedestrian*')
        #lights_list = actors.filter("*traffic_light*")
        objects_list = [all_vehicles,all_walkers]
        for objs in objects_list:
            for vehicle in objs:
                # print(vehicle.id)
                if isinstance(vehicle,carla.libcarla.Walker):
                    scale = 5
                else:
                    scale = 2

                if vehicle.id in keep_vehicle_ids:
                    vehicle_mask = np.zeros([self._width, self._width], dtype=np.uint8)
                    index = keep_vehicle_ids.index(vehicle.id)
                    # cmap = plt.get_cmap('YlOrRd')
                    # c = cmap(object[1])
                    # color = carla.Color(*[int(i*255) for i in c])
                    c = self.get_color(keep_vehicle_attn[index])
                    #color = carla.Color(r=int(c[0]), g=int(c[1]), b=int(c[2]))
                    color = int(c[0]), int(c[1]), int(c[2])
                    loc = vehicle.get_location()
                    bb_loc = carla.Location()
                    bb = carla.BoundingBox(loc, vehicle.bounding_box.extent)
                    actor_transform = carla.Transform(bb.location, vehicle.get_transform().rotation)
                    bb_ext = carla.Vector3D(vehicle.bounding_box.extent)
                    if scale is not None:
                        bb_ext = bb_ext * scale
                        bb_ext.x = max(bb_ext.x, 0.8)
                        bb_ext.y = max(bb_ext.y, 0.8)
                    corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
                               carla.Location(x=bb_ext.x, y=-bb_ext.y),
                               carla.Location(x=bb_ext.x, y=0),
                               carla.Location(x=bb_ext.x, y=bb_ext.y),
                               carla.Location(x=-bb_ext.x, y=bb_ext.y)]
                    corners = [bb_loc + corner for corner in corners]

                    corners = [actor_transform.transform(corner) for corner in corners]
                    corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])
                    corners_warped = cv.transform(corners_in_pixel, M_warp)
                    cv.fillConvexPoly(vehicle_mask, np.round(corners_warped).astype(np.int32), 1)
                    vehicle_mask = vehicle_mask.astype(np.bool)
                    index_mask = vehicle_mask * (1 - real_vehicle_masks[0])
                    attention_mask[index_mask>0] = (int(255*keep_vehicle_attn[index]),0,0) #color #
        #cv2.imwrite("attention_mask.png",attention_mask)
        return attention_mask


    def get_color(self, attention):
        attention = 1
        colors = [
            (255, 255, 255, 255),
            # (220, 228, 180, 255),
            # (190, 225, 150, 255),
            (240, 240, 210, 255),
            # (190, 219, 96, 255),
            (240, 220, 150, 255),
            # (170, 213, 79, 255),
            (240, 210, 110, 255),
            # (155, 206, 62, 255),
            (240, 200, 70, 255),
            # (162, 199, 44, 255),
            (240, 190, 30, 255),
            # (170, 192, 20, 255),
            (240, 185, 0, 255),
            # (177, 185, 0, 255),
            (240, 181, 0, 255),
            # (184, 177, 0, 255),
            (240, 173, 0, 255),
            # (191, 169, 0, 255),
            (240, 165, 0, 255),
            # (198, 160, 0, 255),
            (240, 156, 0, 255),
            # (205, 151, 0, 255),
            (240, 147, 0, 255),
            # (212, 142, 0, 255),
            (240, 137, 0, 255),
            # (218, 131, 0, 255),
            (240, 126, 0, 255),
            # (224, 120, 0, 255),
            (240, 114, 0, 255),
            # (230, 108, 0, 255),
            (240, 102, 0, 255),
            # (235, 95, 0, 255),
            (240, 88, 0, 255),
            # (240, 80, 0, 255),
            (242, 71, 0, 255),
            # (244, 61, 0, 255),
            (246, 49, 0, 255),
            # (247, 34, 0, 255),
            (248, 15, 0, 255),
            (249, 6, 6, 255),
        ]

        ix = int(attention * (len(colors) - 1))
        return colors[ix]
    def clean(self):
        self._parent_actor = None
        self._world = None
        self._history_queue.clear()
    
    def convert_bbox2mask(self,detected_bbox):
        mask = np.zeros((self._width, self._width)).astype(np.uint8)
        for bbox in detected_bbox:
            # Your bounding box parameters
            center_x, center_y = bbox[0], bbox[1]  # center coordinates
            width, height = bbox[4], bbox[5]  # width and height
            angle = bbox[3]

            # Create a rotated rectangle
            rect = ((center_x, center_y), (width, height), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Draw the filled rotated rectangle
            cv2.drawContours(mask, [box], 0, (255, 0, 0), -1)

            """# Calculate the top left corner
            top_left_x = int(center_x - width / 2)
            top_left_y = int(center_y - height / 2)

            # Draw a filled rectangle
            cv2.rectangle(mask, (top_left_x, top_left_y), (top_left_x + width, top_left_y + height),
                          (255, 0, 0), -1)"""

        return mask.astype(np.bool)