import math
import numpy as np
import carla
import cv2 as cv

class Destroy_Actor:
    def __init__(self):
        self._width = 200
        self._distance_threshold = 50.0
        self._pixels_per_meter = 4.0
        self._world_offset = [-470.74255, -252.38815]
        self.prev_loc = None
        self.saved_count = 0

    def _world_to_pixel(self, location, projective=False):
        """Converts the world coordinates to pixel coordinates"""
        x = self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self._pixels_per_meter * (location.y - self._world_offset[1])

        if projective:
            p = np.array([x, y, 1], dtype=np.float32)
        else:
            p = np.array([x, y], dtype=np.float32)
        return p

    def _get_mask_from_actor_list(self, actor_list, M_warp):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        corner_list = []
        for actor_transform, bb_loc, bb_ext in actor_list:

            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=-bb_ext.y),
                       #carla.Location(x=bb_ext.x, y=0),
                       carla.Location(x=bb_ext.x, y=bb_ext.y),
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]
            corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])
            corners_warped = cv.transform(corners_in_pixel, M_warp)
            corner_list.append(corners_warped)

            cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
        return mask.astype(np.bool), corner_list

    def _get_surrounding_actors(self, bbox_list, ev_loc,  scale=None):
        actors = []
        for bbox in bbox_list:
            is_within_distance = self.is_within_distance(bbox, ev_loc)
            if is_within_distance:
                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bbox.extent)
                if scale is not None:
                    bb_ext = bb_ext * scale
                    bb_ext.x = max(bb_ext.x, 0.8)
                    bb_ext.y = max(bb_ext.y, 0.8)

                actors.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
        return actors

    def is_within_distance(self, w, ev_loc):
        c_distance = abs(ev_loc.x - w.location.x) < self._distance_threshold \
                     and abs(ev_loc.y - w.location.y) < self._distance_threshold \
                     and abs(ev_loc.z - w.location.z) < 8.0
        c_ev = abs(ev_loc.x - w.location.x) < 1.0 and abs(ev_loc.y - w.location.y) < 1.0
        return c_distance and (not c_ev)

    def __call__(self, actor_list, lane_mask, M_warp, ev_loc,ego_id, log_saved, constructioncone_list, current_speed):
        if self.prev_loc == None:
            self.prev_loc = ev_loc

        if log_saved and current_speed < 0.5:#ev_loc.distance(self.prev_loc) < 0.1:
            self.saved_count += 1
        elif log_saved:
            self.saved_count = 1

        if self.saved_count%5==0:
            self.saved_count = 1
            lane_mask = lane_mask.astype(np.uint8)
            destroy_actor_list = []
            for actor, bbox, id  in actor_list:#constructioncone_list
                bbox = self._get_surrounding_actors([bbox], ev_loc, scale=1.0)
                if len(bbox) != 0 and id != ego_id:
                    current_vehicle_mask, _ = self._get_mask_from_actor_list(bbox, M_warp)#'vehicle.lincoln.mkz_2020'
                    if np.sum(current_vehicle_mask.astype(np.uint8) * lane_mask) > 0:
                        destroy_actor_list.append(actor)

            for cons in constructioncone_list:
                if cons.get_location().distance(ev_loc) < 5.0:
                    destroy_actor_list.append(cons)



            for actor in destroy_actor_list:
                try:
                    actor.destroy()
                except:
                    pass

        self.prev_loc = ev_loc
