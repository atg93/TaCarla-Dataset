import numpy as np
import cv2 as cv
import os
import pickle
COLOR_WHITE = (255, 255, 255)


class HdMapRasterizer():
    def __init__(self, map):
        self.bev_size = (200, 200)
        self.pixels_per_meter = 4
        self.pixels_ev_to_bottom = 100
        self.distance_threshold = 100 #self.pixels_ev_to_bottom / self.pixels_per_meter
        self.precision = 1.0
        self.carla_map = map
        self.world_offset = np.array([-102.05991, -102.04996])
        topology = [x[0] for x in self.carla_map.get_topology()]
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        self.all_wps = self.extract_wps_from_topology(topology, precision=self.precision)
        #self.road_line_pickle = os.getenv('TAIRDRIVE_ROOT') + '/tools/leaderboard_tools/hd_map_pickles/town12_roads_lines.pickle'
        self.road_line_pickle = '/home/tg22/remote-pycharm/tairdrive'+ '/tools/leaderboard_tools/hd_map_pickles/town12_roads_lines.pickle'
        self.road_line_pickle = '/tools_0/leaderboard_tools/hd_map_pickles/town12_roads_lines.pickle'
        self.all_polygons, self.all_left_marks, self.all_right_marks = self.init_wps(self.all_wps) #dictionary of waypoints of the same road id

    def init_wps(self, all_wps):
        if os.path.exists(self.road_line_pickle):
            with open(self.road_line_pickle, 'rb') as f:
                data = pickle.load(f)
            all_polygons = data['all_polygons']
            all_left_marks = data['all_left_marks']
            all_right_marks = data['all_right_marks']
        else:
            all_polygons = []
            all_left_marks = []
            all_right_marks = []
            for waypoints in all_wps:

                lane_left_side = [HdMapRasterizer.lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
                lane_right_side = [HdMapRasterizer.lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]
                polygon = lane_left_side + [x for x in reversed(lane_right_side)]
                polygon = [HdMapRasterizer.world_to_pixel(x, self.pixels_per_meter, self.world_offset) for x in polygon]

                all_polygons.append(polygon)
                mark_left_side = [HdMapRasterizer.world_to_pixel(
                    HdMapRasterizer.lateral_shift(w.transform, -w.lane_width * 0.5),
                    self.pixels_per_meter, self.world_offset) for w in waypoints if not w.is_junction]
                mark_right_side = [HdMapRasterizer.world_to_pixel(
                    HdMapRasterizer.lateral_shift(w.transform, w.lane_width * 0.5),
                    self.pixels_per_meter, self.world_offset) for w in waypoints if not w.is_junction]
                all_left_marks.append(mark_left_side)
                all_right_marks.append(mark_right_side)

        return all_polygons, all_left_marks, all_right_marks

    def extract_wps_from_topology(self, topology, precision):
        all_wps = []
        for waypoint in topology:
            waypoints = [waypoint]
            # Generate waypoints of a road id. Stop when road id differs
            nxt = waypoint.next(precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                while nxt.road_id == waypoint.road_id:
                    waypoints.append(nxt)
                    nxt = nxt.next(precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break
            all_wps.append(waypoints)
        return all_wps

    def rasterize_hd_map(self, actor_transform):

        m_warp = self.get_warp_transform(actor_transform.location, actor_transform.rotation, pixels_per_meter=self.pixels_per_meter)

        drivable_area, lines, lane_dict = self.draw_drivable_line(self.all_wps, m_warp, actor_transform)
        #cv.imwrite("lines.png", lines * 255)
        return drivable_area, lines, lane_dict

    def check_distance(self, wp, actor_transform):
        return (np.abs(wp.transform.location.x - actor_transform.location.x) < self.distance_threshold and
         np.abs(wp.transform.location.y - actor_transform.location.y) < self.distance_threshold)

    def check_every_100th_waypoint(self, waypoints, actor_transform):
        array_length = len(waypoints)
        if array_length < 100:
            if array_length > 0:
                # Check the first waypoint
                if self.check_distance(waypoints[0], actor_transform):
                    return True
                # Check the last waypoint
                if self.check_distance(waypoints[-1], actor_transform):
                    return True
            return False

        for i in range(0, len(waypoints), 100):
            if self.check_distance(waypoints[i], actor_transform):
                return True
        return False

    def draw_drivable_line(self, all_wps, M_warp, actor_transform):
        road_surface = np.zeros((200,200))
        lane_marking_all_surface = np.zeros((200,200))



        lane_dict = {}
        for polygon_index, waypoints in enumerate(all_wps):
            if self.check_every_100th_waypoint(waypoints, actor_transform):
                self.draw_road(road_surface, polygon_index, COLOR_WHITE, M_warp)
                points_left, points_right = self.draw_line(lane_marking_all_surface, polygon_index, 1, M_warp)
                if len(points_left) != 0 and len(points_right) != 0:
                    midpoint = (points_left + points_right) / 2
                    lane_dict.update({polygon_index:{'midpoint':midpoint, "points_left":points_left, "points_right":points_right}})
                    asd = 0

        lane_vector = np.zeros((200, 200, 3))
        for sample_key in lane_dict.keys():
            sample = lane_dict[sample_key]
            lane_midpoint_mask = np.zeros((200, 200))
            lane_points_left_mask = np.zeros((200, 200))
            lane_points_right_mask = np.zeros((200, 200))
            cv.polylines(lane_points_left_mask, [sample['points_left']], False, 1,
                         thickness=1)  # np.expand_dims(sample['midpoint'],1)
            cv.polylines(lane_points_right_mask, [sample['points_right']], False, 1,
                         thickness=1)  # np.expand_dims(sample['midpoint'],1)
            cv.polylines(lane_midpoint_mask, np.array(sample['midpoint'], dtype=np.int32).reshape((1, -1, 2)), False, 1,
                         thickness=1)  # np.expand_dims(sample['midpoint'],1)
            lane_vector[lane_midpoint_mask > 0] = (0, 255, 0)  # b,g,r
            lane_vector[lane_points_left_mask > 0] = (255, 0, 0)
            lane_vector[lane_points_right_mask > 0] = (0, 0, 255)
        #cv.imwrite("lane_vector.png", lane_vector)
        return road_surface, lane_marking_all_surface, lane_dict

    def get_warp_transform(self,ev_loc, ev_rot, pixels_per_meter):
        pixels_ev_to_bottom = 100
        width = 200
        yaw = np.deg2rad(ev_rot.yaw)

        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        right_vec = np.array([np.cos(yaw + 0.5*np.pi), np.sin(yaw + 0.5*np.pi)])

        ev_loc_in_px = HdMapRasterizer.world_to_pixel(ev_loc, pixels_per_meter, self.world_offset)
        bottom_left = ev_loc_in_px - pixels_ev_to_bottom * forward_vec - (0.5*width) * right_vec
        top_left = ev_loc_in_px + (width-pixels_ev_to_bottom) * forward_vec - (0.5*width) * right_vec
        top_right = ev_loc_in_px + (width-pixels_ev_to_bottom) * forward_vec + (0.5*width) * right_vec

        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.array([[0, width-1],
                            [0, 0],
                            [width-1, 0]], dtype=np.float32)

        return cv.getAffineTransform(src_pts, dst_pts)

    def transform_to_bev(self, M_warp, waypoints):
        lane_points = cv.transform(np.array([[waypoints]])[0], M_warp)

        return lane_points[0]

    @staticmethod
    def world_to_pixel(location, pixels_per_meter, world_offset):
        """Converts the world coordinates to pixel coordinates"""
        x = pixels_per_meter * (location.x - world_offset[0])
        y = pixels_per_meter * (location.y - world_offset[1])
        return [round(x), round(y)] #because pygame takes it y , x

    @staticmethod
    def lateral_shift(transform, shift):
        """Makes a lateral shift of the forward vector of a transform"""
        transform.rotation.yaw += 90
        return transform.location + shift * transform.get_forward_vector()

    def draw_road(self, surface, polygon_index, color, M_warp):
        """Renders a single lane in a surface and with a specified color"""
        polygon = self.transform_to_bev(M_warp, self.all_polygons[polygon_index])
        if len(polygon) > 2:
            cv.fillPoly(surface, [polygon], color)

    def draw_line(self, surface, polygon_index, width, M_warp):
        """Draws solid lines in a surface given a set of points, width and color"""
        points_left, points_right = [], []
        if len(self.all_left_marks[polygon_index]) >= 2:
            points_left = self.transform_to_bev(M_warp, self.all_left_marks[polygon_index])
            cv.polylines(surface, [points_left], False, 1, thickness=width)
        if len(self.all_right_marks[polygon_index]) >= 2:
            points_right = self.transform_to_bev(M_warp, self.all_right_marks[polygon_index])
            cv.polylines(surface, [points_right], False, 1, thickness=width)

        return points_left, points_right
