import numpy as np
import carla
from gym import spaces
from collections import deque
from pathlib import Path


import os
import cv2


class Compute_Prediction_Input():
    def __init__(self, _width, _pixels_per_meter,_pixels_ev_to_bottom, chauffeurnet):
        self._ego_motion_queue = deque(maxlen=3)
        self._width = _width
        self._pixels_per_meter = _pixels_per_meter
        self._pixels_ev_to_bottom = _pixels_ev_to_bottom
        self.chauffeurnet = chauffeurnet

    def __call__(self, ev_loc, ev_rot, ev_transform, ev_bbox, obs_dict,vehicle_masks):
        prev_vehicle_masks, warp_ego_motion = self._warp(ev_loc, ev_rot, ev_transform, ev_bbox)
        self.update_obs(obs_dict,warp_ego_motion,prev_vehicle_masks,vehicle_masks)

    def _get_ego_motion(self):
        ev_transform = self.chauffeurnet._parent_actor.vehicle.get_transform()
        ev_loc = np.array([ev_transform.location.x, ev_transform.location.y, ev_transform.location.z])
        ev_rot = np.array([ev_transform.rotation.pitch, ev_transform.rotation.yaw, ev_transform.rotation.roll])

        return np.concatenate((ev_loc, ev_rot), axis=0).reshape(2, 3)

    def update_obs(self,obs_dict,warp_ego_motion,prev_vehicle_masks,vehicle_masks):
        obs_dict.update({'warp_ego_motion': warp_ego_motion})
        obs_dict.update({'prediction_input': prev_vehicle_masks})
        obs_dict.update({'vehicle_masks': vehicle_masks})
        return obs_dict

    def _warp(self, ev_loc, ev_rot, ev_transform, ev_bbox):
        self._ego_motion_queue.append((ev_loc, ev_rot, ev_transform, ev_bbox))

        if len(self._ego_motion_queue) == 3:
            index = 2
            M_warp_previous = self.chauffeurnet._get_warp_transform(self._ego_motion_queue[index][0],
                                                       self._ego_motion_queue[index][1])

            # objects with history
            prev_vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks \
                = self.chauffeurnet._get_history_masks(M_warp_previous)

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
