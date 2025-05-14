import copy

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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import h5py
import json

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


def fake_tint(color, factor):
    r = color
    return r


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
        self._masks_channels = 23  # 3 + 3*len(self._history_idx) #23
        self._parent_actor = None
        self._world = None
        self.chan_w_history_per_actor = self._image_channels + 1

        self._map_dir = Path(__file__).resolve().parent / 'maps'

        self.orginal_route_design = True
        self.vehicle_dic_dataset = {}
        self.vehicle_index_dataset = 1
        self.not_moving_index = 0
        self.not_moving_actor_id = 0
        self.not_moving_actor_dic = {}

        self._history_queue_dataset = deque(maxlen=20)

        self.save_ego_pixel_cor = False
        self.dataset_folder_name = 'datasets'
        self.frame_number = 0
        self.is_not_moving_vec_considered = True

        self.dataset_save_step = 0
        self.dataset_save_step_freq = 5

        self.data_collection_sample_number = 0

        if not os.path.exists(self.dataset_folder_name):
            os.makedirs(self.dataset_folder_name)

        ####dataset label
        self.collect_dataset = True
        self.add_instance_id_model = False
        self.save_image = False
        self.vehicle_non_seen = False

        self.episode_number = 0

        self.open_simulation = False
        self.instance_input = False
        self.route_plan_from_DQN = None

        # self.reset_episode()

        super(ObsManager, self).__init__()

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
            walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance, 2.0)
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
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks \
            = self._get_history_masks(M_warp)

        # road_mask, lane_mask
        road_mask = cv.warpAffine(self._road, M_warp, (self._width, self._width)).astype(np.bool)
        lane_mask_all = cv.warpAffine(self._lane_marking_all, M_warp, (self._width, self._width)).astype(np.bool)
        lane_mask_broken = cv.warpAffine(self._lane_marking_white_broken, M_warp,
                                         (self._width, self._width)).astype(np.bool)

        # route_mask
        if self.route_plan_from_DQN != None:
            self._parent_actor.route_plan[0:80] = self.route_plan_from_DQN

        if self.orginal_route_design:
            route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
            route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                       for wp, _ in self._parent_actor.route_plan[0:80]])
            route_warped = cv.transform(route_in_pixel, M_warp)
            cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=16)
            route_mask = route_mask.astype(np.bool)
        else:
            ###all lanes
            route_mask = np.zeros([self._width, self._width], dtype=np.uint8)

            self.new_parent_actor_route_plan, self.new_parent_actor_route_plan_dic, lane_number_list = self.get_all_lane_route_plan(
                self._parent_actor.route_plan[0:80])
            route_in_pixel = np.array([[self._world_to_pixel(wp_transform.location)]
                                       for wp_transform in self.new_parent_actor_route_plan])

            route_warped = cv.transform(route_in_pixel, M_warp)

            wp_tuple = self._parent_actor.route_plan[0]
            wp, _ = wp_tuple
            for index, r_w in enumerate(route_warped):
                if index != 0:
                    thickness = int(wp.lane_width * 4.5 * lane_number_list[0])
                    new_wrap = [r_w, old_rw]
                    cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=thickness)
                old_rw = r_w

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

        """if self.instance_input:
            vehicle_masks, walker_masks, _, _, _, _, _, _, _, _, _ = self.creating_instance_input()
            c_walker_history = walker_masks
            c_vehicle_history = vehicle_masks
            #plt.imsave("deneme_vehicle.png", np.array(c_vehicle_history[0]).reshape(self._width, self._width))
            #plt.imsave("deneme_walker.png", np.array(c_walker_history[0]).reshape(self._width, self._width))


        else:
            c_vehicle_history = [m*255 for m in vehicle_masks]
            c_walker_history = [m*255 for m in walker_masks]"""

        c_vehicle_history = [m * 255 for m in vehicle_masks]
        c_walker_history = [m * 255 for m in walker_masks]

        if self.add_instance_id_model:
            vehicle_masks_instance_id, walker_masks_instance_id, _, _, _, _, _, _, _, _, _ = self.creating_instance_input()
            c_walker_history_instance_id = walker_masks_instance_id
            c_vehicle_history_instance_id = vehicle_masks_instance_id
            assert np.array(c_vehicle_history).shape == np.array(c_vehicle_history_instance_id).shape
            assert np.array(c_walker_history).shape == np.array(c_walker_history_instance_id).shape

            masks = np.stack((c_road, c_route, c_lane, *c_vehicle_history, *c_walker_history, *c_tl_history,
                              *c_vehicle_history_instance_id, *c_walker_history_instance_id), axis=2)

        else:
            if self.vehicle_non_seen:
                c_vehicle_history = [m * 0 for m in vehicle_masks]
            masks = np.stack((c_road, c_route, c_lane, *c_vehicle_history, *c_walker_history, *c_tl_history), axis=2)

        masks = np.transpose(masks, [2, 0, 1])

        obs_dict = {'rendered': image, 'masks': masks}

        if self.save_image and self.orginal_route_design:
            cv2.imwrite('image.png', image)
        elif self.save_image:
            cv2.imwrite('all_route_image.png', image)

        self._parent_actor.collision_px = np.any(ev_mask_col & walker_masks[-1])

        return obs_dict

    def _get_history_masks(self, M_warp):
        qsize = len(self._history_queue)
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks = [], [], [], [], [], []
        for idx in self._history_idx:
            idx = max(idx, -1 * qsize)

            vehicles, walkers, tl_green, tl_yellow, tl_red, stops = self._history_queue[idx]
            # print("self._history_queue:",self._history_queue)
            vehicle_masks.append(self._get_mask_from_actor_list_vec_ped(vehicles, M_warp))
            walker_masks.append(self._get_mask_from_actor_list_vec_ped(walkers, M_warp))
            tl_green_masks.append(self._get_mask_from_stopline_vtx(tl_green, M_warp))
            tl_yellow_masks.append(self._get_mask_from_stopline_vtx(tl_yellow, M_warp))
            tl_red_masks.append(self._get_mask_from_stopline_vtx(tl_red, M_warp))
            stop_masks.append(self._get_mask_from_actor_list(stops, M_warp))

        return vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks

    def _get_history_masks_datasets(self, M_warp):
        qsize = len(self._history_queue_dataset)
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks = [], [], [], [], [], []

        for idx in self._history_idx:
            idx = max(idx, -1 * qsize)

            vehicles, walkers, tl_green, tl_yellow, tl_red, stops = self._history_queue_dataset[idx]
            vehicle_masks.append(self._get_mask_from_actor_list_for_dataset(vehicles, M_warp))
            walker_masks.append(self._get_mask_from_actor_list_for_dataset(walkers, M_warp))
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

    def _get_mask_from_actor_list_vec_ped(self, actor_list, M_warp):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        for bbox, actor_transform, bb_loc, bb_ext in actor_list:
            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),  # bottom_left
                       carla.Location(x=bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=0),
                       carla.Location(x=bb_ext.x, y=bb_ext.y),
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]
            corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])
            corners_warped = cv.transform(corners_in_pixel, M_warp)
            cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)

        # print("np.round(corners_warped).astype(np.int32):", np.round(corners_warped).astype(np.int32))

        return mask.astype(np.bool)

    def _get_mask_from_actor_list(self, actor_list, M_warp):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        for actor_transform, bb_loc, bb_ext in actor_list:
            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),  # bottom_left
                       carla.Location(x=bb_ext.x, y=-bb_ext.y),  # top_left
                       carla.Location(x=bb_ext.x, y=bb_ext.y),  # top_right
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]  # bottom_right
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]
            corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])

            corners_warped = cv.transform(corners_in_pixel, M_warp)
            cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)

        # print("np.round(corners_warped).astype(np.int32):", np.round(corners_warped).astype(np.int32))

        return mask.astype(np.bool)

    def _get_pixel_cor_from_actor_list(self, actor_list, M_warp):
        for actor_transform, bb_loc, bb_ext in actor_list:
            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),  # bottom_left
                       carla.Location(x=bb_ext.x, y=-bb_ext.y),  # top_left
                       carla.Location(x=bb_ext.x, y=bb_ext.y),  # top_right
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]  # bottom_right
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]
            corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])

            corners_warped = cv.transform(corners_in_pixel, M_warp)
            # print("corners_warped:",corners_warped)

        pixel_corner_dict = {}
        keys = ["bottom_left", "top_left", "top_right", "bottom_right"]
        for index, cor in enumerate(corners_warped):
            pixel_corner_dict.update({keys[index]: cor[0].tolist()})
        return pixel_corner_dict

    def _get_mask_from_actor_list_for_dataset(self, actor_list, M_warp):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)

        actor_id_list = []
        actor_type_id_list = []
        mask_list = []
        self.reset_id_dic()
        actor_max_index = 0
        for actor_id, actor_type_id, actor_transform, bb_loc, bb_ext in actor_list:
            mask_init = np.zeros([self._width, self._width], dtype=np.uint8)

            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=0),
                       carla.Location(x=bb_ext.x, y=bb_ext.y),
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]

            corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])
            corners_warped = cv.transform(corners_in_pixel, M_warp)

            if self.can_actor_be_seen(corners_warped):
                index = self.get_id_index(actor_id)
                actor_max_index += 1
            else:
                index = -1
            actor_id_list.append(index)
            actor_type_id_list.append(actor_type_id)

            cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
            cv.fillConvexPoly(mask_init, np.round(corners_warped).astype(np.int32), 1)

            mask_list.append(mask_init.astype(np.bool))
        actor_max_index = 30  # max(1,actor_max_index)
        return actor_id_list, actor_type_id_list, mask_list, mask.astype(np.bool), actor_max_index

    @staticmethod
    def _get_surrounding_actors(bbox_list, criterium, scale=None):
        actors = []
        for bbox in bbox_list:

            if type(bbox) == carla.BoundingBox:
                is_within_distance = criterium(bbox)
            else:
                is_within_distance, bbox = criterium(bbox)

            if is_within_distance:
                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bbox.extent)
                if scale is not None:
                    bb_ext = bb_ext * scale
                    bb_ext.x = max(bb_ext.x, 0.8)
                    bb_ext.y = max(bb_ext.y, 0.8)
                actors.append((bbox, carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
        return actors

    @staticmethod
    def _get_surrounding_actors_for_datasets(actor_list, criterium, scale=None):
        actors = []
        for ac in actor_list:

            is_within_distance, bbox_word_location, bbox_rotation, bbox = criterium(ac)

            if is_within_distance:
                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bbox.extent)
                if scale is not None:
                    bb_ext = bb_ext * scale
                    bb_ext.x = max(bb_ext.x, 0.8)
                    bb_ext.y = max(bb_ext.y, 0.8)

                new_location = carla.Location(bbox.location.x + bbox_word_location.x,
                                              bbox.location.y + bbox_word_location.y,
                                              bbox.location.z + bbox_word_location.z)
                actors.append((ac.id, ac.type_id, carla.Transform(new_location, bbox_rotation), bb_loc, bb_ext))

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

    def get_average_wp(self, wp_list):
        sum_x = 0
        sum_y = 0
        sum_z = 0
        for wp in wp_list:
            sum_x += wp.transform.location.x
            sum_y += wp.transform.location.y
            sum_z += wp.transform.location.z
        return sum_x / len(wp_list), sum_y / len(wp_list), sum_z / len(wp_list)

    def get_all_lane_route_plan(self, route_plan):

        new_route_plan = []
        new_route_dic = {}
        lane_number_list = []
        for index, rt in enumerate(route_plan):
            wp, _ = rt
            wp_list = [wp]

            wp_right = wp
            lane_type = wp_right.lane_type
            wp_count = 0
            orginal_rotation = wp.transform.rotation
            while wp_right != None and lane_type == carla.LaneType.Driving and wp_right.transform.rotation == orginal_rotation:
                wp_right = wp_right.get_right_lane()
                if wp_right != None:
                    lane_type = wp_right.lane_type
                    if lane_type == carla.LaneType.Driving and wp_right.transform.rotation == orginal_rotation:
                        wp_list.append(wp_right)
                if wp_count > 5:
                    break
                wp_count += 1

            wp_list.reverse()

            wp_left = wp
            lane_type = wp_left.lane_type
            wp_count = 0
            while wp_left != None and lane_type == carla.LaneType.Driving and wp_left.transform.rotation == orginal_rotation:
                wp_left = wp_left.get_left_lane()
                if wp_left != None:
                    lane_type = wp_left.lane_type
                    if lane_type == carla.LaneType.Driving and wp_left.transform.rotation == orginal_rotation:
                        wp_list.append(wp_left)

                if wp_count > 5:
                    break
                wp_count += 1

            average_loc = self.get_average_wp(wp_list)
            new_location = carla.Location(x=average_loc[0], y=average_loc[1], z=average_loc[2])
            wp_transform = carla.Transform(new_location, wp.transform.rotation)
            new_route_plan.append(wp_transform)
            lane_number_list.append(len(wp_list))

            """lane_number = 0
            for lane_numbar, wp in enumerate(wp_list):
                new_location = carla.Location(x=wp.transform.location.x, y=wp.transform.location.y, z=wp.transform.location.z)
                wp_transform = carla.Transform(new_location, wp.transform.rotation)
                if index==0:
                    new_route_dic.update({lane_number:[wp_transform]})
                else:
                    new_route_dic[lane_number].append(wp_transform)

                lane_number += 1"""

        return new_route_plan, new_route_dic, lane_number_list

    def collect_data_for_future_prediction(self):

        outputs = self.creating_instance_input()
        vehicle_image_data, pedestrian_image_data, road_image, c_road_mask, c_lane, c_tl_history, image, ev_mask, ev_transform, ev_bbox, M_warp = outputs

        if self.collect_dataset:
            if self.dataset_save_step % self.dataset_save_step_freq == 0:
                self.save_arrays(vehicle_image_data, pedestrian_image_data, road_image, c_road_mask, c_lane,
                                 c_tl_history)
                self.data_collection_sample_number += 1
                if self.save_ego_pixel_cor:
                    ego_pixel_dic = self._get_pixel_cor_from_actor_list(
                        [(ev_transform, ev_bbox.location, ev_bbox.extent)],
                        M_warp)
                    print("ego_pixel_dic:", ego_pixel_dic)
                    with open('ego_pixel_cor.json', 'w') as json_file:
                        json.dump(ego_pixel_dic, json_file)
                    self.save_ego_pixel_cor = False
            self.dataset_save_step += 1

        if self.open_simulation:
            image[ev_mask] = COLOR_WHITE

            cv2.imwrite('fake_image.png', image)

        self.frame_number += 1

        return vehicle_image_data, pedestrian_image_data

    def creating_instance_input(self):
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

        def fake_is_within_distance(vehicle):
            if type(vehicle) == carla.BoundingBox:
                w = vehicle
                bbox = w
                fake_ev_loc = ev_loc
            else:
                w = vehicle.get_transform()
                bbox = vehicle.bounding_box
                fake_ev_loc = self._parent_actor.vehicle.get_transform().location
            c_distance = abs(fake_ev_loc.x - w.location.x) < self._distance_threshold \
                         and abs(fake_ev_loc.y - w.location.y) < self._distance_threshold \
                         and abs(fake_ev_loc.z - w.location.z) < 8.0
            c_ev = abs(fake_ev_loc.x - w.location.x) < 1.0 and abs(fake_ev_loc.y - w.location.y) < 1.0

            return c_distance and (not c_ev), w.location, w.rotation, bbox

        vehicle_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Vehicles)
        walker_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Pedestrians)

        vehicle_actors = self._world.get_actors().filter('*vehicle*')
        walker_actors = self._world.get_actors().filter('*walker.pedestrian*')
        actor_bounding_box = []

        for vec in vehicle_actors:
            actor_bounding_box.append(vec.bounding_box)

        if self._scale_bbox:
            get_level_vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance, 1.0)
            get_level_walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance, 2.0)
        else:
            get_level_vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance)
            get_level_walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance)

        if self._scale_bbox:
            vehicles = self._get_surrounding_actors_for_datasets(vehicle_actors, fake_is_within_distance, 1.0)
            walkers = self._get_surrounding_actors_for_datasets(walker_actors, fake_is_within_distance, 2.0)
        else:
            vehicles = self._get_surrounding_actors_for_datasets(vehicle_actors, fake_is_within_distance)
            walkers = self._get_surrounding_actors_for_datasets(walker_actors, fake_is_within_distance)

        if len(get_level_vehicles) != len(vehicles) and self.is_not_moving_vec_considered:
            vehicles = self.creating_id_for_not_moving_vec(1.0, get_level_vehicles, vehicles)

        # print("after not moving vehicles:",vehicles)

        tl_green = TrafficLightHandler.get_stopline_vtx(ev_loc, 0)
        tl_yellow = TrafficLightHandler.get_stopline_vtx(ev_loc, 1)
        tl_red = TrafficLightHandler.get_stopline_vtx(ev_loc, 2)
        stops = self._get_stops(self._parent_actor.criteria_stop)

        self._history_queue_dataset.append((vehicles, walkers, tl_green, tl_yellow, tl_red, stops))

        M_warp = self._get_warp_transform(ev_loc, ev_rot)

        # objects with history
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks \
            = self._get_history_masks_datasets(M_warp)

        # road_mask, lane_mask
        road_mask = cv.warpAffine(self._road, M_warp, (self._width, self._width)).astype(np.bool)
        lane_mask_all = cv.warpAffine(self._lane_marking_all, M_warp, (self._width, self._width)).astype(np.bool)
        lane_mask_broken = cv.warpAffine(self._lane_marking_white_broken, M_warp,
                                         (self._width, self._width)).astype(np.bool)

        # route_mask
        route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                   for wp, _ in self._parent_actor.route_plan[0:80]])
        route_warped = cv.transform(route_in_pixel, M_warp)
        cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=16)
        route_mask = route_mask.astype(np.bool)

        # ev_mask
        ev_mask = self._get_mask_from_actor_list_vec_ped([(ev_bbox, ev_transform, ev_bbox.location, ev_bbox.extent)],
                                                         M_warp)
        ev_mask_col = self._get_mask_from_actor_list_vec_ped([(ev_bbox, ev_transform, ev_bbox.location,
                                                               ev_bbox.extent * self._scale_mask_col)], M_warp)

        # render
        image = np.zeros([self._width, self._width, 3], dtype=np.uint8)
        image[road_mask] = COLOR_ALUMINIUM_5
        image[lane_mask_all] = COLOR_MAGENTA
        image[lane_mask_broken] = COLOR_MAGENTA_2
        road_image = copy.deepcopy(image.reshape(1, self._width, self._width, 3))
        image[route_mask] = COLOR_ALUMINIUM_3

        h_len = len(self._history_idx) - 1
        for i, mask in enumerate(stop_masks):
            image[mask] = tint(COLOR_YELLOW_2, (h_len - i) * 0.2)
        for i, mask in enumerate(tl_green_masks):
            image[mask] = tint(COLOR_GREEN, (h_len - i) * 0.2)
        for i, mask in enumerate(tl_yellow_masks):
            image[mask] = tint(COLOR_YELLOW, (h_len - i) * 0.2)
        for i, mask in enumerate(tl_red_masks):
            image[mask] = tint(COLOR_RED, (h_len - i) * 0.2)

        ####
        # masks
        c_road_mask = road_mask * 255
        c_lane = lane_mask_all * 255
        c_lane[lane_mask_broken] = 255

        # masks with history
        c_tl_history = []
        for i in range(len(self._history_idx)):
            c_tl = np.zeros([self._width, self._width], dtype=np.uint8)
            c_tl[tl_green_masks[i]] = 80
            c_tl[tl_yellow_masks[i]] = 170
            c_tl[tl_red_masks[i]] = 255
            c_tl[stop_masks[i]] = 255
            c_tl_history.append(c_tl)
        c_tl_history = [c_tl_history[3]]

        c_lane = np.array(c_lane).reshape(1, self._width, self._width)
        c_tl_history = np.array(c_tl_history).reshape(1, self._width, self._width)
        c_road_mask = np.array(c_road_mask).reshape(1, self._width, self._width)
        ####

        if not self.instance_input:
            image_vehicle_masks = [vehicle_masks[3]]
        else:
            image_vehicle_masks = vehicle_masks

        vehicle_list = []
        for i, mask in enumerate(image_vehicle_masks):
            mod_actor_id, actor_type, mask_list, mask, actor_max_index = mask
            image[mask] = tint(COLOR_BLUE, (h_len - i) * 0.2)

            int_image = np.zeros([self._width, self._width, 1], dtype=np.uint8)
            for index, mask_sample in enumerate(mask_list):
                if not self.instance_input:
                    int_image[mask_sample] = mod_actor_id[index]
                else:
                    int_image[mask_sample] = (mod_actor_id[index] / (actor_max_index)) * 255
            vehicle_list.append(int_image)

        if not self.instance_input:
            vehicle_image_data = np.array(vehicle_list).reshape(1, self._width, self._width)
        else:
            vehicle_image_data = np.array(vehicle_list).reshape(self.chan_w_history_per_actor, self._width, self._width)
        # name = 'frame'+str(self.frame_number)+'_vec_image_pixel.png'
        # plt.imsave(name, np.array(vehicle_image_data[0]).reshape(self._width, self._width))
        # plt.imsave(name, np.array(vehicle_image_data[0]).reshape(self._width, self._width))

        vanilla_vehicle_masks = []
        for vec in vehicle_masks:
            vanilla_vehicle_masks.append(vec[3])
        vehicle_masks = vanilla_vehicle_masks

        if not self.instance_input:
            image_walker_masks = [walker_masks[3]]
        else:
            image_walker_masks = walker_masks

        walker_list = []
        for i, mask in enumerate(image_walker_masks):
            mod_actor_id, actor_type, mask_list, mask, actor_max_index = mask
            image[mask] = tint(COLOR_BLUE, (h_len - i) * 0.2)

            int_image = np.zeros([self._width, self._width, 1], dtype=np.uint8)
            for index, mask_sample in enumerate(mask_list):
                if not self.instance_input:
                    int_image[mask_sample] = mod_actor_id[index]
                else:
                    int_image[mask_sample] = (mod_actor_id[index] / (actor_max_index)) * 255

            walker_list.append(int_image)
        if not self.instance_input:
            pedestrian_image_data = np.array(walker_list).reshape(1, self._width, self._width)
        else:
            pedestrian_image_data = np.array(walker_list).reshape(self.chan_w_history_per_actor, self._width,
                                                                  self._width)

        return vehicle_image_data, pedestrian_image_data, road_image, c_road_mask, c_lane, c_tl_history, \
               image, ev_mask, ev_transform, ev_bbox, M_warp

    def creating_id_for_not_moving_vec(self, scale, get_level_vehicles, vehicles):
        for get_level_v in get_level_vehicles:
            not_moving_label = True
            for vec in vehicles:
                distance = get_level_v[1].location.distance(vec[2].location)
                if distance < 1:
                    not_moving_label = False
                    break

            if not_moving_label:
                key = str(get_level_v[1].location.x) + '_' + str(get_level_v[1].location.y)
                bbox = get_level_v[0]
                if key in self.not_moving_actor_dic.keys():
                    fake_actor_id = self.not_moving_actor_dic[key]
                else:
                    self.not_moving_actor_id -= 1
                    fake_actor_id = self.not_moving_actor_id

                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bbox.extent)
                if scale is not None:
                    bb_ext = bb_ext * scale
                    bb_ext.x = max(bb_ext.x, 0.8)
                    bb_ext.y = max(bb_ext.y, 0.8)

                self.not_moving_actor_dic.update({key: fake_actor_id})

                vehicles.append((self.not_moving_actor_id, "not_moving_vehicle", get_level_v[1], bb_loc, bb_ext))
        return vehicles

    def save_arrays(self, vehicle_image, pedesterians_image, road_image, c_road_mask, c_lane, c_tl_history):

        ego_motion = self._get_ego_motion()
        if self.counter == 0:
            self.h5_file.create_dataset("vehicles", data=np.expand_dims(vehicle_image[0], axis=0),
                                        chunks=True, maxshape=(None, self._width, self._width))
            self.h5_file.create_dataset("pedesterians",
                                        data=np.expand_dims(pedesterians_image[0], axis=0),
                                        chunks=True, maxshape=(None, self._width, self._width))
            self.h5_file.create_dataset("road_image",
                                        data=np.expand_dims(road_image[0], axis=0),
                                        chunks=True, maxshape=(None, self._width, self._width, 3))
            self.h5_file.create_dataset("ego_motion",
                                        data=np.expand_dims(ego_motion, axis=0),
                                        chunks=True, maxshape=(None, 2, 3))
            self.h5_file.create_dataset("tl",
                                        data=np.expand_dims(c_tl_history[0], axis=0),
                                        chunks=True, maxshape=(None, self._width, self._width))
            self.h5_file.create_dataset("road_mask",
                                        data=np.expand_dims(c_road_mask[0], axis=0),
                                        chunks=True, maxshape=(None, self._width, self._width))
            self.h5_file.create_dataset("lane",
                                        data=np.expand_dims(c_lane[0], axis=0),
                                        chunks=True, maxshape=(None, self._width, self._width))



        else:  # append next arrays to dataset
            self.h5_file["vehicles"].resize((self.h5_file["vehicles"].shape[0] + 1), axis=0)
            self.h5_file["vehicles"][-1] = vehicle_image[0]
            self.h5_file["pedesterians"].resize((self.h5_file["pedesterians"].shape[0] + 1), axis=0)
            self.h5_file["pedesterians"][-1] = pedesterians_image[0]
            self.h5_file["road_image"].resize((self.h5_file["road_image"].shape[0] + 1), axis=0)
            self.h5_file["road_image"][-1] = road_image
            self.h5_file["ego_motion"].resize((self.h5_file["ego_motion"].shape[0] + 1), axis=0)
            self.h5_file["ego_motion"][-1] = ego_motion
            self.h5_file["tl"].resize((self.h5_file["tl"].shape[0] + 1), axis=0)
            self.h5_file["tl"][-1] = c_tl_history[0]
            self.h5_file["road_mask"].resize((self.h5_file["road_mask"].shape[0] + 1), axis=0)
            self.h5_file["road_mask"][-1] = c_tl_history[0]
            self.h5_file["lane"].resize((self.h5_file["lane"].shape[0] + 1), axis=0)
            self.h5_file["lane"][-1] = c_tl_history[0]

        self.counter += 1

    def _get_ego_motion(self):
        ev_transform = self._parent_actor.vehicle.get_transform()
        ev_loc = np.array([ev_transform.location.x, ev_transform.location.y, ev_transform.location.z])
        ev_rot = np.array([ev_transform.rotation.pitch, ev_transform.rotation.yaw, ev_transform.rotation.roll])

        return np.concatenate((ev_loc, ev_rot), axis=0).reshape(2, 3)

    def reset_episode(self):

        self.counter = 0
        self.vehicle_dic_dataset = {}
        self.vehicle_index_dataset = 1

        name = self.dataset_folder_name + '/episode_' + str(self.episode_number).zfill(3) + '.h5'
        self.h5_file = h5py.File(self.dataset_folder_name + '/episode_' + str(self.episode_number).zfill(3) + '.h5',
                                 'w')

    def close_h5_file(self):
        self.h5_file.close()

    def set_episode_number(self, task_idx):
        self.episode_number = task_idx

    def set_collect_data_info(self, data_collection):
        self.data_collection = data_collection

    def update_route_plan(self, new_route_plan):
        self.route_plan_from_DQN = new_route_plan

    def get_ego_and_wp_list(self):
        ev_transform = self._parent_actor.vehicle.get_transform()
        ev_loc = ev_transform.location
        return ev_loc, self._parent_actor.route_plan[0:80]

    def reset_id_dic(self):
        if not self.collect_dataset:
            self.vehicle_dic_dataset = {}
            self.vehicle_index_dataset = 1

    def can_actor_be_seen(self, corners_warped):
        corner_vectors = corners_warped.reshape(5, -1)
        be_seen = False
        for cor in corner_vectors:
            if np.all(cor > 0) and np.all(cor < 192):
                be_seen = True
                break
        return be_seen

    def get_id_index(self, actor_id):
        if actor_id not in self.vehicle_dic_dataset.keys():
            self.vehicle_dic_dataset.update({actor_id: self.vehicle_index_dataset})
            index = self.vehicle_index_dataset
            self.vehicle_index_dataset += 1
        else:
            index = self.vehicle_dic_dataset[actor_id]

        return index

    def set_orginal_route_label(self, status):
        self.orginal_route_design = status






