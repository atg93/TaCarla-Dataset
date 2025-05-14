import numpy as np
import carla
from gym import spaces
import cv2 as cv
from collections import deque
from pathlib import Path
import os

print("chaff",os.getcwd())
from carla_gym.core.obs_manager.obs_manager import ObsManagerBase
from carla_gym.core.obs_manager.birdview.compute_prediction_input import Compute_Prediction_Input

from carla_gym.utils.traffic_light import TrafficLightHandler
from carla_gym.core.obs_manager.birdview.planning.planning import Planning

import h5py
import cv2

import math
import pyproj
from carla_gym.core.obs_manager.birdview.gps_warped import Gps_Warped

#from geopy.distance import VincentyDistance
from geopy.distance import geodesic as GD
import geopy.distance

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

        self._image_channels = 3
        self._masks_channels = 3 + 3*len(self._history_idx)
        self._parent_actor = None
        self._world = None
        self.collect_VAE_data = False

        self._map_dir = Path(__file__).resolve().parent / 'maps'

        super(ObsManager, self).__init__()

        self.compute_prediction_input = None #Compute_Prediction_Input(self._width, self._pixels_per_meter,self._pixels_ev_to_bottom, self)
        self.planning = Planning()
        self._ego_motion_queue = deque(maxlen=3)
        self.sensor = None

    def _define_obs_space(self):
        self.obs_space = spaces.Dict(
            {'rendered': spaces.Box(
                low=0, high=255, shape=(self._width, self._width, self._image_channels),
                dtype=np.uint8),
             'masks': spaces.Box(
                low=0, high=255, shape=(self._masks_channels, self._width, self._width),
                dtype=np.uint8)})

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor
        self._world = self._parent_actor.vehicle.get_world()

        maps_h5_path = self._map_dir / (self._world.get_map().name + '.h5')
        with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf:
            self._road = np.array(hf['road'], dtype=np.uint8)
            self._lane_marking_all = np.array(hf['lane_marking_all'], dtype=np.uint8)
            self._lane_marking_white_broken = np.array(hf['lane_marking_white_broken'], dtype=np.uint8)
            # self._shoulder = np.array(hf['shoulder'], dtype=np.uint8)
            # self._parking = np.array(hf['parking'], dtype=np.uint8)
            # self._sidewalk = np.array(hf['sidewalk'], dtype=np.uint8)
            # self._lane_marking_yellow_broken = np.array(hf['lane_marking_yellow_broken'], dtype=np.uint8)
            # self._lane_marking_yellow_solid = np.array(hf['lane_marking_yellow_solid'], dtype=np.uint8)
            # self._lane_marking_white_solid = np.array(hf['lane_marking_white_solid'], dtype=np.uint8)

            self._world_offset = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)
            assert np.isclose(self._pixels_per_meter, float(hf.attrs['pixels_per_meter']))

        self._distance_threshold = np.ceil(self._width / self._pixels_per_meter)
        self.gps_warped = None#Gps_Warped(image_width=self._width, pixels_per_meter=self._pixels_per_meter, _world_offset=self._world_offset, _history_idx=self._history_idx)

        # dilate road mask, lbc draw road polygon with 10px boarder
        # kernel = np.ones((11, 11), np.uint8)
        # self._road = cv.dilate(self._road, kernel, iterations=1)

    @staticmethod
    def _get_stops(criteria_stop):
        stop_sign = criteria_stop._target_stop_sign
        stops = []
        if (stop_sign is not None) and (not criteria_stop._stop_completed):
            bb_loc = carla.Location(stop_sign.trigger_volume.location)
            bb_ext = carla.Vector3D(stop_sign.trigger_volume.extent)
            bb_ext.x = max(bb_ext.x, bb_ext.y)
            bb_ext.y = max(bb_ext.x, bb_ext.y)
            trans = stop_sign.get_transform()
            stops = [(carla.Transform(trans.location, trans.rotation), bb_loc, bb_ext)]
        return stops

    def get_observation(self):
        ev_transform = self._parent_actor.vehicle.get_transform()
        ev_loc = ev_transform.location
        ev_rot = ev_transform.rotation
        ev_bbox = self._parent_actor.vehicle.bounding_box
        snap_shot = self._world.get_snapshot()

        def is_within_distance(w):
            c_distance = abs(ev_loc.x - w.location.x) < self._distance_threshold \
                and abs(ev_loc.y - w.location.y) < self._distance_threshold \
                and abs(ev_loc.z - w.location.z) < 8.0
            c_ev = abs(ev_loc.x - w.location.x) < 1.0 and abs(ev_loc.y - w.location.y) < 1.0
            return c_distance and (not c_ev)

        vehicle_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Vehicles)
        walker_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Pedestrians)
        if self._scale_bbox:
            vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance, 1.0)
            walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance, 3.0)
        else:
            vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance)
            walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance)

        tl_green = TrafficLightHandler.get_stopline_vtx(ev_loc, 0)
        tl_yellow = TrafficLightHandler.get_stopline_vtx(ev_loc, 1)
        tl_red = TrafficLightHandler.get_stopline_vtx(ev_loc, 2)
        stops = self._get_stops(self._parent_actor.criteria_stop)

        self._history_queue.append((vehicles, walkers, tl_green, tl_yellow, tl_red, stops))

        M_warp = self._get_warp_transform(ev_loc, ev_rot)

        # objects with history
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks, vec_mask_coordinate, vec_mask_world_coordinate, vec_mask_object_word_cartesian \
            = self._get_history_masks(M_warp)


        # road_mask, lane_mask
        road_mask = cv.warpAffine(self._road, M_warp, (self._width, self._width)).astype(np.bool)
        lane_mask_all = cv.warpAffine(self._lane_marking_all, M_warp, (self._width, self._width)).astype(np.bool)
        lane_mask_broken = cv.warpAffine(self._lane_marking_white_broken, M_warp,
                                         (self._width, self._width)).astype(np.bool)

        planning_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        #route plannig
        control = None
        if isinstance(self.sensor,dict):
            control, pred_wp = self.planning(world=self._world, _global_plan=self._parent_actor.route_plan, ego_actor=self._parent_actor, _input=self.sensor)
            pred_wp = [93, 100] - pred_wp.detach().cpu().numpy().reshape((4, 2))
            for _cor in pred_wp:
                planning_mask = cv2.circle(planning_mask, (int(_cor[1]), int(_cor[0])), radius=1, color=(255), thickness=-1)
            pred_wp = np.flip(pred_wp, axis=None)

            #cv2.imwrite("planning_mask.png", planning_mask)
            #sum_wp = np.sum(planning_mask)

        # route_mask
        route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                   for wp, _ in self._parent_actor.route_plan[0:80]])
        route_warped = cv.transform(route_in_pixel, M_warp)
        try:
            cv.polylines(route_mask, [np.round(pred_wp).astype(np.int32)], False, 1, thickness=16)
        except:
            cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=16)
        route_mask = route_mask.astype(np.bool)

        # ev_mask
        ev_mask, ev_mask_pixel, ev_mask_world_coordinate, ev_mask_object_word_cartesian = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location, ev_bbox.extent)], M_warp)
        ev_mask_col, ev_mask_col_pixel, ev_mask_col_world_coordinate, ev_mask_object_word_cartesian = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location,
                                                       ev_bbox.extent*self._scale_mask_col)], M_warp)
        deneme_image = np.zeros([self._width, self._width, 3], dtype=np.uint8)
        deneme_image[road_mask] = COLOR_ALUMINIUM_5
        deneme_image[route_mask] = COLOR_ALUMINIUM_3
        deneme_image[lane_mask_all] = COLOR_MAGENTA
        deneme_image[lane_mask_broken] = COLOR_MAGENTA_2

        if self.sensor != None and type(self.gps_warped) != type(None):
            self.gps_warped.add_que(bounding_boxes_list=vec_mask_coordinate, vec_mask_world_coordinate=vec_mask_world_coordinate, vec_mask_object_word_cartesian=vec_mask_object_word_cartesian,ego_pixel_coordinate=ev_mask_col_pixel, ego_loc=ev_mask_col_world_coordinate, compass=self.sensor['sensor'][1]['compass'])

            pixel_coordinates, deneme_vec_masks = self.gps_warped.get_que_item(ego_pixel_coordinate=ev_mask_col_pixel, ego_loc=ev_mask_col_world_coordinate, ev_mask_object_word_cartesian=ev_mask_object_word_cartesian, compass=self.sensor['sensor'][1]['compass'])
            deneme_image[deneme_vec_masks>0] = COLOR_BLUE

        deneme_image[ev_mask] = COLOR_WHITE

        # render
        image = np.zeros([self._width, self._width, 3], dtype=np.uint8)
        image[road_mask] = COLOR_ALUMINIUM_5
        image[route_mask] = COLOR_ALUMINIUM_3
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

        for i, mask in enumerate(vehicle_masks):
            image[mask] = tint(COLOR_BLUE, (h_len-i)*0.2)
        for i, mask in enumerate(walker_masks):
            image[mask] = tint(COLOR_CYAN, (h_len-i)*0.2)

        image[ev_mask] = COLOR_WHITE
        
        image[planning_mask>0] = COLOR_RED
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
            c_tl[tl_green_masks[i]] = 0 #80
            c_tl[tl_yellow_masks[i]] = 170
            c_tl[tl_red_masks[i]] = 255
            c_tl[stop_masks[i]] = 255
            c_tl_history.append(c_tl)

        c_vehicle_history = [m*255 for m in vehicle_masks]
        c_walker_history = [m*255 for m in walker_masks]

        masks = np.stack((c_road, c_route, c_lane, *c_vehicle_history, *c_walker_history, *c_tl_history), axis=2)
        masks = np.transpose(masks, [2, 0, 1])

        #prev_vehicle_masks, warp_ego_motion = self._warp(ev_loc, ev_rot, ev_transform, ev_bbox)
        obs_dict = {'rendered': image, 'masks': masks,'plant_control':control}

        if self.compute_prediction_input != None:
            self.compute_prediction_input(ev_loc, ev_rot, ev_transform, ev_bbox, obs_dict, vehicle_masks)


        self._parent_actor.collision_px = np.any(ev_mask_col & walker_masks[-1])

        return obs_dict

    def _get_history_masks(self, M_warp):
        qsize = len(self._history_queue)
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks = [], [], [], [], [], []
        for idx in self._history_idx:
            idx = max(idx, -1 * qsize)

            vehicles, walkers, tl_green, tl_yellow, tl_red, stops = self._history_queue[idx]

            vec_mask, vec_mask_coordinate, vec_mask_world_coordinate, vec_mask_object_word_cartesian = self._get_mask_from_actor_list(vehicles, M_warp)
            vehicle_masks.append(vec_mask)
            walker_masks.append(self._get_mask_from_actor_list(walkers, M_warp)[0])
            tl_green_masks.append(self._get_mask_from_stopline_vtx(tl_green, M_warp)[0])
            tl_yellow_masks.append(self._get_mask_from_stopline_vtx(tl_yellow, M_warp)[0])
            tl_red_masks.append(self._get_mask_from_stopline_vtx(tl_red, M_warp)[0])
            stop_masks.append(self._get_mask_from_actor_list(stops, M_warp)[0])

        return vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks, vec_mask_coordinate, vec_mask_world_coordinate, vec_mask_object_word_cartesian

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
        corners_warped_list = []
        object_word_coordinate = []
        object_word_cartesian = []
        for actor_transform, bb_loc, bb_ext in actor_list:

            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y), #bottom_left
                       carla.Location(x=bb_ext.x, y=-bb_ext.y), #top_left
                       carla.Location(x=bb_ext.x, y=0),
                       carla.Location(x=bb_ext.x, y=bb_ext.y), #top_right
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)] #bottom_right
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]
            corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])
            corners_warped = cv.transform(corners_in_pixel, M_warp)

            if corners_warped.mean(0)[0][0] > 0 and corners_warped.mean(0)[0][0] < self._width and corners_warped.mean(0)[0][1] > 0 and corners_warped.mean(0)[0][1] < self._width:
                orientation = -1
                if type(self.gps_warped) != type(None):
                    orientation = self.gps_warped.calculate_orientation(top_left_x=corners_warped[1][0][0], top_left_y=corners_warped[1][0][1], bottom_left_x=corners_warped[0][0][0], bottom_left_y=corners_warped[0][0][1])
                width, height = self.calculate_width_height(corners=corners_warped)
                x = corners_warped.mean(0)[0][0]
                y = corners_warped.mean(0)[0][1]
                corners_warped_list.append((np.round(np.array([x, y])),np.array([height, width]),orientation))
                self._world.get_map().transform_to_geolocation(actor_transform.location)
                gps_coor = self._world.get_map().transform_to_geolocation(actor_transform.location)
                object_word_cartesian.append(np.array([actor_transform.location.x, actor_transform.location.y, actor_transform.location.z*(-1)]))
                object_word_coordinate.append(np.array([gps_coor.latitude,gps_coor.longitude,gps_coor.altitude]))
            cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
        return mask.astype(np.bool), corners_warped_list, object_word_coordinate, object_word_cartesian

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
        try:
            x = location.x
            y = location.y
        except:
            x = location[0]
            y = location[1]

        """Converts the world coordinates to pixel coordinates"""
        x = self._pixels_per_meter * (x - self._world_offset[0])
        y = self._pixels_per_meter * (y - self._world_offset[1])

        if projective:
            p = np.array([x, y, 1], dtype=np.float32)
        else:
            p = np.array([x, y], dtype=np.float32)
        return p

    def _pixel_to_world(self, location, projective=False):
        x = location[0]
        y = location[1]

        """Converts the world coordinates to pixel coordinates"""
        x = (x/self._pixels_per_meter) + self._world_offset[0]
        y = (y/self._pixels_per_meter) + self._world_offset[1]

        if projective:
            p = np.array([x, y, 1], dtype=np.float32)
        else:
            p = np.array([x, y], dtype=np.float32)
        return p


    def _world_to_pixel_width(self, width):
        """Converts the world units to pixel units"""
        return self._pixels_per_meter * width

    def calculate_width_height(self, corners):
        corners = corners.reshape(5, 2)
        # Convert corner points to NumPy array for easier calculations
        corners_array = np.array(corners)

        # Calculate the center of the bounding box
        center = corners_array.mean(axis=0)

        # Compute covariance matrix
        cov_matrix = np.cov(corners_array, rowvar=False)

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Find the index of the largest eigenvalue
        largest_eigenvalue_index = np.argmax(eigenvalues)

        # Use the corresponding eigenvector to determine the orientation
        orientation_vector = eigenvectors[:, largest_eigenvalue_index]

        # Compute width and height
        width = np.linalg.norm(np.dot(corners_array - center, orientation_vector))
        height = np.linalg.norm(np.dot(corners_array - center, orientation_vector[::-1]))

        return int(round(width)), int(round(height))

    def set_sensor(self, sensors):
        # gps = np.array([sensors[0]['lat'], sensors[0]['lon']])#['compass']
        self.sensor = {'gps': np.array([sensors[0]['lat'], sensors[0]['lon']]), 'imu': sensors[1], 'sensor': sensors}

    def clean(self):
        self._parent_actor = None
        self._world = None
        self._history_queue.clear()
