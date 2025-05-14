import json

import numpy as np
import carla
from gym import spaces
import cv2 as cv
from collections import deque
from pathlib import Path
import h5py

from carla_gym.core.obs_manager.obs_manager import ObsManagerBase
from carla_gym.utils.traffic_light import TrafficLightHandler

import cv2

import os

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
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
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
        #self._scale_bbox = True
        self._scale_mask_col = obs_configs.get('scale_mask_col', 1.1)

        self._history_queue = deque(maxlen=20)

        self._ego_motion_queue = deque(maxlen=3)
        self.__queue = deque(maxlen=3)
        self._ego_motion_queue = deque(maxlen=3)

        self._image_channels = 3
        self._masks_channels = 3 + 3 * len(self._history_idx)
        self._parent_actor = None
        self._world = None

        self._map_dir = Path(__file__).resolve().parent / 'maps'

        #######future_prediction
        self.future_vec_prediciton = False
        if self.future_vec_prediciton:
            self._masks_channels += 1

        self.prev_ev_transform = None
        self.prev_ev_bbox_location = None
        self.prev_ev_bbox_extent = None
        self.future_prediction = None
        self.trafic_light_disable = False
        self.save_ego_pixel_location = False
        self.collect_VAE_data = False

        self.maximum_x = -1
        self.maximum_y = -1

        super(ObsManager, self).__init__()

        if self.collect_VAE_data:
            self.dataset_folder_name = 'VAE_datasets'
            if not os.path.exists(self.dataset_folder_name):
                os.makedirs(self.dataset_folder_name)

            self.reset_episode()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict(
            {'rendered': spaces.Box(
                low=0, high=255, shape=(self._width, self._width, self._image_channels),
                dtype=np.uint8),
                'masks': spaces.Box(
                    low=0, high=255, shape=(self._masks_channels, self._width, self._width),
                    dtype=np.uint8),
                'e_brake_input': spaces.Box(
                    low=0, high=255, shape=(9, self._width, self._width),
                    dtype=np.uint8)
            })  # tugrul

    def set_trafic_light_disable(self, trafic_light_disable):
        self.trafic_light_disable = trafic_light_disable

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
            vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance, 1.0, object='vehicle')
            walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance, 3.0, object='walker')
        else:
            vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance)
            walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance, object='walker')

        tl_green = TrafficLightHandler.get_stopline_vtx(ev_loc, 0)
        tl_yellow = TrafficLightHandler.get_stopline_vtx(ev_loc, 1)
        tl_red = TrafficLightHandler.get_stopline_vtx(ev_loc, 2)
        stops = self._get_stops(self._parent_actor.criteria_stop)

        self._history_queue.append((vehicles, walkers, tl_green, tl_yellow, tl_red, stops))

        M_warp = self._get_warp_transform(ev_loc, ev_rot)

        # objects with history
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks \
            = self._get_history_masks(M_warp)

        # vec_masks_list = self._get_vec_masks_and_id(M_warp)

        # road_mask, lane_mask
        road_mask = cv.warpAffine(self._road, M_warp, (self._width, self._width)).astype(np.bool)
        lane_mask_all = cv.warpAffine(self._lane_marking_all, M_warp, (self._width, self._width)).astype(np.bool)
        lane_mask_broken = cv.warpAffine(self._lane_marking_white_broken, M_warp,
                                         (self._width, self._width)).astype(np.bool)

        # for intersection
        index_array = np.arange(1, 5) * 2
        self.route_mask_list = []
        for index in index_array:
            route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
            route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                       for wp, _ in self._parent_actor.route_plan[0:index]])
            route_warped = cv.transform(route_in_pixel, M_warp)
            cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=1)
            route_mask = route_mask.astype(np.bool)
            self.route_mask_list.append(route_mask)

        self.route_mask_of_e_brake_light = self.create_route_mask(M_warp, start_index=0, end_index=15, thickness=1)
        self.route_image_of_e_brake = self.create_route_mask(M_warp, start_index=2, end_index=10, thickness=5)

        self.route_mask = route_mask
        self.vehicle_masks = vehicle_masks[1:]
        self.walker_masks = walker_masks[1:]

        # route_mask
        route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                   for wp, _ in self._parent_actor.route_plan[0:80]])
        route_warped = cv.transform(route_in_pixel, M_warp)
        cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=12)
        route_mask = route_mask.astype(np.bool)

        # ev_mask
        ev_mask = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location, ev_bbox.extent)], M_warp)
        ev_mask_col = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location,
                                                       ev_bbox.extent * self._scale_mask_col)], M_warp)

        # render
        image = np.zeros([self._width, self._width, 3], dtype=np.uint8)
        image[road_mask] = COLOR_ALUMINIUM_5
        image[route_mask] = COLOR_ALUMINIUM_3
        image[lane_mask_all] = COLOR_MAGENTA
        image[lane_mask_broken] = COLOR_MAGENTA_2

        h_len = len(self._history_idx) - 1
        # tugrul
        if self.trafic_light_disable:
            stop_masks = np.zeros((4, self._width, self._width)) > 0
            tl_green_masks = np.zeros((4, self._width, self._width)) > 0
            tl_yellow_masks = np.zeros((4, self._width, self._width)) > 0
            tl_red_masks = np.zeros((4, self._width, self._width)) > 0

        for i, mask in enumerate(stop_masks):
            image[mask] = tint(COLOR_YELLOW_2, (h_len - i) * 0.2)
        for i, mask in enumerate(tl_green_masks):
            image[mask] = tint(COLOR_GREEN, (h_len - i) * 0.2)
        for i, mask in enumerate(tl_yellow_masks):
            image[mask] = tint(COLOR_YELLOW, (h_len - i) * 0.2)
        for i, mask in enumerate(tl_red_masks):
            image[mask] = tint(COLOR_RED, (h_len - i) * 0.2)

        for i, mask in enumerate(vehicle_masks):
            image[mask] = tint(COLOR_BLUE, (h_len - i) * 0.2)

        for i, mask in enumerate(walker_masks):
            image[mask] = tint(COLOR_CYAN, (h_len - i) * 0.2)

        image[ev_mask] = COLOR_WHITE
        # image[obstacle_mask] = COLOR_BLUE

        ego_vehicle_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        ego_vehicle_mask[ev_mask] = 255

        # masks
        c_road = road_mask * 255
        c_route = route_mask * 255
        c_lane = lane_mask_all * 255
        c_lane[lane_mask_broken] = 120

        # masks with history
        c_tl_history = []
        for i in range(len(self._history_idx)):
            c_tl = np.zeros([self._width, self._width], dtype=np.uint8)
            c_tl[tl_green_masks[i]] = 0  # 80
            c_tl[tl_yellow_masks[i]] = 170
            c_tl[tl_red_masks[i]] = 255
            c_tl[stop_masks[i]] = 255
            c_tl_history.append(c_tl)

        c_vehicle_history = [m * 255 for m in vehicle_masks]
        c_walker_history = [m * 255 for m in walker_masks]

        masks = np.stack((c_road, c_route, c_lane, *c_vehicle_history, *c_walker_history, *c_tl_history), axis=2)

        masks = np.transpose(masks, [2, 0, 1])

        ego_motion = self._get_ego_motion()

        prev_vehicle_masks, warp_ego_motion = self._warp(ev_loc, ev_rot, ev_transform, ev_bbox)

        e_brake_input = self.get_e_brake_obs_and_route_info(c_route, c_walker_history, c_vehicle_history, c_tl_history)

        self.current_vehicles = c_vehicle_history[-1]
        self.current_walker = c_walker_history[-1]
        self.current_red_light = tl_red_masks[-1]

        is_there_tl = np.sum(np.multiply(self.route_mask_of_e_brake_light, self.current_red_light)) > 0
        is_there_intersection_tl = np.sum(np.multiply(ego_vehicle_mask, self.current_red_light)) > 0
        is_there_vec = np.sum(self.current_vehicles) > 0

        obs_dict = {'rendered': image, 'masks': masks, 'current_vehicles': c_vehicle_history[-1],
                    'current_walker': c_walker_history[-1],
                    'terminate_route_image_of_e_brake': self.route_image_of_e_brake,
                    'intersection_route_mask_list': self.route_mask_list,
                    'intersection_vehicle_masks': self.vehicle_masks, 'intersection_walker_masks': self.walker_masks,
                    'e_brake_input': e_brake_input, 'warp_ego_motion': warp_ego_motion,
                    'prediction_input': prev_vehicle_masks, 'c_vehicle_history': c_vehicle_history,
                    'c_tl_history': c_tl_history, 'c_walker_history': c_walker_history, 'vehicle_masks': vehicle_masks,
                    'is_there_tl': is_there_tl,
                    'is_there_intersection_tl': is_there_intersection_tl,
                    'is_there_vec': is_there_vec}
        self._parent_actor.collision_px = np.any(ev_mask_col & walker_masks[-1])

        cv2.imwrite("image_0_o.png",image)
        # cv2.imwrite("ego_vehicle_mask.png",ego_vehicle_mask)
        if self.save_ego_pixel_location:
            ego_pixel_dic = self._get_pixel_cor_from_actor_list([(ev_transform, ev_bbox.location, ev_bbox.extent)],
                                                                M_warp)
            print("ego_pixel_dic:", ego_pixel_dic)
            with open('ego_pixel_cor.json', 'w') as json_file:
                json.dump(ego_pixel_dic, json_file)

        if self.collect_VAE_data:
            self.save_arrays(masks)

        return obs_dict

    def _get_history_masks(self, M_warp):
        qsize = len(self._history_queue)
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks = [], [], [], [], [], []
        for idx in self._history_idx:
            idx = max(idx, -1 * qsize)

            vehicles, walkers, tl_green, tl_yellow, tl_red, stops = self._history_queue[idx]

            vehicle_masks.append(self._get_mask_from_actor_list(vehicles, M_warp))
            walker_masks.append(self._get_mask_from_actor_list(walkers, M_warp))
            tl_green_masks.append(self._get_mask_from_stopline_vtx(tl_green, M_warp))
            tl_yellow_masks.append(self._get_mask_from_stopline_vtx(tl_yellow, M_warp))
            tl_red_masks.append(self._get_mask_from_stopline_vtx(tl_red, M_warp))
            stop_masks.append(self._get_mask_from_actor_list(stops, M_warp))

        return vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks

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
            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),#bottom_left
                       carla.Location(x=bb_ext.x, y=-bb_ext.y), #top_left
                       carla.Location(x=bb_ext.x, y=0),#center
                       carla.Location(x=bb_ext.x, y=bb_ext.y),#top_right
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]#bottom_right
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]
            corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])
            corners_warped = cv.transform(corners_in_pixel, M_warp)

            cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
        return mask.astype(np.bool)

    @staticmethod
    def _get_surrounding_actors(bbox_list, criterium, scale=None, object='vehicle'):
        actors = []
        for bbox in bbox_list:
            is_within_distance = criterium(bbox)
            if is_within_distance:
                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bbox.extent)
                if scale is not None:
                    """dist = np.linalg.norm(np.array([bb_ext.x, bb_ext.y]))
                    if object == 'vehicle' and dist < 1.5:
                        bb_ext.x *= 2.5
                        bb_ext.y *= 2.0
                    else:"""
                    bb_ext = bb_ext * scale

                    bb_ext.x = max(bb_ext.x, 1.0)
                    bb_ext.y = max(bb_ext.y, 1.0)

                actors.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
        return actors

    def _get_warp_transform(self, ev_loc, ev_rot):
        ev_loc_in_px = self._world_to_pixel(ev_loc)
        yaw = np.deg2rad(ev_rot.yaw)

        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        right_vec = np.array([np.cos(yaw + 0.5 * np.pi), np.sin(yaw + 0.5 * np.pi)])

        bottom_left = ev_loc_in_px - self._pixels_ev_to_bottom * forward_vec - (0.5 * self._width) * right_vec
        top_left = ev_loc_in_px + (self._width - self._pixels_ev_to_bottom) * forward_vec - (
                    0.5 * self._width) * right_vec
        top_right = ev_loc_in_px + (self._width - self._pixels_ev_to_bottom) * forward_vec + (
                    0.5 * self._width) * right_vec

        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.array([[0, self._width - 1],
                            [0, 0],
                            [self._width - 1, 0]], dtype=np.float32)
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

    def clean(self):
        self._parent_actor = None
        self._world = None
        self._history_queue.clear()

    def _get_ego_motion(self):
        ev_transform = self._parent_actor.vehicle.get_transform()
        ev_loc = np.array([ev_transform.location.x, ev_transform.location.y, ev_transform.location.z])
        ev_rot = np.array([ev_transform.rotation.pitch, ev_transform.rotation.yaw, ev_transform.rotation.roll])

        return np.concatenate((ev_loc, ev_rot), axis=0).reshape(2, 3)

    def _warp(self, ev_loc, ev_rot, ev_transform, ev_bbox):
        self._ego_motion_queue.append((ev_loc, ev_rot, ev_transform, ev_bbox))

        if len(self._ego_motion_queue) == 3:
            index = 2
            M_warp_previous = self._get_warp_transform(self._ego_motion_queue[index][0],
                                                       self._ego_motion_queue[index][1])

            # objects with history
            prev_vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks \
                = self._get_history_masks(M_warp_previous)

            # tugrul
            prev_vehicle_masks = np.array(prev_vehicle_masks) + np.array(walker_masks)

            ev_loc = np.array([self._ego_motion_queue[index][0].x, self._ego_motion_queue[index][0].y,
                               self._ego_motion_queue[index][0].z])
            ev_rot = np.array([self._ego_motion_queue[index][1].pitch, self._ego_motion_queue[index][1].yaw,
                               self._ego_motion_queue[index][1].roll])

            return prev_vehicle_masks[1:], np.concatenate((ev_loc, ev_rot), axis=0).reshape(2, 3)
        else:
            prev_vehicle_masks = list(np.zeros((3, self._width, self._width)))
            ego_motion = self._get_ego_motion()

            return prev_vehicle_masks, ego_motion

    def is_there_intersection(self, prediction_output, intersection_with="both"):

        if intersection_with == "vehicle":
            dynamic_masks = self.vehicle_masks
        elif intersection_with == "walker":
            dynamic_masks = self.walker_masks
        else:
            dynamic_masks = (np.array(self.walker_masks) + np.array(self.vehicle_masks)) > 0

        dynamic_image = np.zeros((self._width, self._width, 1))
        for d_masks in dynamic_masks:
            dynamic_image[d_masks] = 1

        intersection_label = False
        index = 0
        intersecrion_sum = 0
        for route_mask in self.route_mask_list:
            route_image = np.zeros((self._width, self._width, 1))
            route_image[route_mask] = 1

            prediction_output = prediction_output.reshape(self._width, self._width, 1)
            prediction_output_mask = prediction_output > 0

            prediction_output[prediction_output_mask] = 1

            prediction_intersection = np.multiply(prediction_output, route_image)

            vehicle_intersection = np.multiply(dynamic_image, route_image)
            index += 1

            if np.sum(prediction_intersection) > 0 and np.sum(vehicle_intersection) == 0:
                intersection_label = True
                intersecrion_sum = np.sum(prediction_intersection)
                break
            elif np.sum(vehicle_intersection) > 0:
                intersecrion_sum = 0
                break
        return intersection_label, intersecrion_sum

    def create_route_mask(self, M_warp, start_index=0, end_index=6, thickness=16):
        route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        input = np.zeros([self._width, self._width], dtype=np.uint8)
        route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                   for wp, _ in self._parent_actor.route_plan[start_index:end_index]])
        route_warped = cv.transform(route_in_pixel, M_warp)
        cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=thickness)
        route_mask = route_mask.astype(np.bool)
        input[route_mask] = 1
        return input

    def check_brake_label(self, prediction_output):
        prediction_output = prediction_output.reshape(self._width, self._width)
        prediction_output_mask = prediction_output > 0
        prediction_output[prediction_output_mask] = 1
        # prediction_intersection = np.multiply(prediction_output, self.route_image_of_e_brake)
        current_obstacle = self.current_walker + self.current_vehicles  # + prediction_output
        return np.sum(np.multiply(self.route_image_of_e_brake, current_obstacle)) > 0 or np.sum(
            np.multiply(self.route_mask_of_e_brake_light, self.current_red_light))

    def get_e_brake_obs_and_route_info(self, route_input, c_walker_history, c_vehicle_history, c_tl_history):
        c_obs_history = list(np.array(c_vehicle_history) + np.array(c_walker_history))
        masks = np.stack((route_input, *c_tl_history, *c_obs_history), axis=2)
        return np.transpose(masks, [2, 0, 1])

    def get_ego_and_wp_list(self):
        ev_transform = self._parent_actor.vehicle.get_transform()
        ev_loc = ev_transform.location
        return ev_loc, self._parent_actor.route_plan[0:80]

    def get_ego_transform(self):
        return self._parent_actor.vehicle.get_transform()

    def get_wp_points(self):
        return self._parent_actor.route_plan

    def reset_episode(self):
        self.counter = 0
        episode_number = 0
        file = self.dataset_folder_name +'/episode_'+str(episode_number).zfill(3)+'.h5'
        file_exist = True

        while file_exist:
            episode_number += 1
            file = self.dataset_folder_name + '/episode_' + str(episode_number).zfill(3) + '.h5'
            file_exist = os.path.exists(file)

        self.h5_file = h5py.File(file, 'w')

    def save_arrays(self,masks):
        if self.counter == 0:
            self.h5_file.create_dataset("masks", data=np.expand_dims(masks, axis=0),
                                        chunks=True, maxshape=(None, self._masks_channels, self._width, self._width))

        else:  # append next arrays to dataset
            self.h5_file["masks"].resize((self.h5_file["masks"].shape[0] + 1), axis=0)
            self.h5_file["masks"][-1] = masks

        self.counter += 1


    def _get_pixel_cor_from_actor_list(self, actor_list, M_warp):
        for actor_transform, bb_loc, bb_ext in actor_list:
            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),#bottom_left
                       carla.Location(x=bb_ext.x, y=-bb_ext.y), #top_left
                       carla.Location(x=bb_ext.x, y=bb_ext.y), #top_right
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]#bottom_right
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]
            corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])

            corners_warped = cv.transform(corners_in_pixel, M_warp)

        pixel_corner_dict = {}
        keys = ["bottom_left","top_left","top_right","bottom_right"]
        for index, cor in enumerate(corners_warped):
            pixel_corner_dict.update({keys[index]:cor[0].tolist()})
        return pixel_corner_dict


