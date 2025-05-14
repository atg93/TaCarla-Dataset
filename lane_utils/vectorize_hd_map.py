import numpy as np
COLOR_WHITE = (255, 255, 255)


class HdMapVectorizer():
    def __init__(self, map):
        self.distance_threshold = 50 
        self.precision = 1.0 #distance of meters between each waypoint
        self.check_N = int(self.distance_threshold / self.precision * 2)
        self.carla_map = map
        self.world_offset = np.array([-102.05991, -102.04996]) #do not change for town 12
        topology = [x[0] for x in self.carla_map.get_topology()]
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        self.all_wps = self.extract_wps_from_topology(topology, precision=self.precision)
        self.all_centerlines, self.all_polygons, self.all_left_marks, self.all_right_marks = self.init_wps(self.all_wps) #dictionary of waypoints of the same road id

    # initialize and constructs all centerlines, polygons, left marks, right marks
    def init_wps(self, all_wps):

        all_polygons = []
        all_left_marks = []
        all_right_marks = []
        all_centerlines = []
        for waypoints in all_wps:

            lane_left_side = [HdMapVectorizer.lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints.copy()]
            lane_right_side = [HdMapVectorizer.lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints.copy()]
            polygon = lane_left_side + [x for x in reversed(lane_right_side)]
            centerline = [[w.transform.location.x, w.transform.location.y, w.transform.location.z] for w in waypoints.copy()]

            all_polygons.append(polygon)
            all_left_marks.append(lane_left_side)
            all_right_marks.append(lane_right_side)
            all_centerlines.append(centerline)

        return all_centerlines, all_polygons, all_left_marks, all_right_marks

    # returns all vectors which are inside the bev boundaries.
    def vectorize_hd_map(self, actor_transform):
        centerline_vectors = []
        polygon_vectors = []
        left_mark_vectors = []
        right_mark_vectors = []

        for vector_index, waypoints in enumerate(self.all_wps):
            if self.check_every_Nth_waypoint(waypoints, actor_transform):
                centerline_vectors.append(self.all_centerlines[vector_index])
                polygon_vectors.append(self.all_polygons[vector_index])
                left_mark_vectors.append(self.all_left_marks[vector_index])
                right_mark_vectors.append(self.all_right_marks[vector_index])

        return centerline_vectors, polygon_vectors, left_mark_vectors, right_mark_vectors

    @staticmethod
    def lateral_shift(transform, shift):
        """Makes a lateral shift of the forward vector of a transform"""
        transform.rotation.yaw += 90
        vector = transform.location + shift * transform.get_forward_vector()
        return [vector.x, vector.y, vector.z]

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

    #this function checks every Nth waypoint distance to the ego. (if precision is 1, and 100m check is enough for you --> N = 1)
    #it returns true if that group of waypoints is inside yopur bev zone (partiallu or fully no matter)
    def check_every_Nth_waypoint(self, waypoints, actor_transform):
        array_length = len(waypoints)
        if array_length < self.check_N:
            if array_length > 0:
                # Check the first waypoint
                if self.check_distance(waypoints[0], actor_transform):
                    return True
                # Check the last waypoint
                if self.check_distance(waypoints[-1], actor_transform):
                    return True
            return False

        for i in range(0, len(waypoints), self.check_N):
            if self.check_distance(waypoints[i], actor_transform):
                return True
        return False

    def check_distance(self, wp, actor_transform):
        return (np.abs(wp.transform.location.x - actor_transform.location.x) < self.distance_threshold and
         np.abs(wp.transform.location.y - actor_transform.location.y) < self.distance_threshold)
