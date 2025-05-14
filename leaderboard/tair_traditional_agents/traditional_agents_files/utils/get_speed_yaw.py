import copy
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import colorsys
import webcolors

class Get_Speed_Yaw:
    def __init__(self):
        self.previous_bb = None
        self.initial_id_count = None
        self.time_count = 0
        self.bbox_dict = {}

    def create_initial_id_to_bbox(self, bbox):
        if isinstance(self.initial_id_count,type(None)):
            self.initial_id_count = 0
        item = {'item':self.time_count,'bbox':[bbox]}
        self.bbox_dict.update({self.initial_id_count:item})
        self.initial_id_count += 1

    def add_prev_bbox(self, key_id, prev_bbox):
        item = {'item': self.time_count, 'bbox': prev_bbox}
        self.bbox_dict.update({key_id: item})

    def select_proper_index(self,array, mean=False):
        array = np.array(array)
        vector = np.abs(array - np.mean(array))
        if mean:
            return np.argmin(vector.mean(-1))
        else:
            return np.argmin(vector)


    def __call__(self,bbox, prev_bbox_list, ego_motion, M_warp_list, compass,obs_image, orientation_angle_list, ego_location=100):
        image = obs_image#np.ones((200, 200, 3), np.uint8) * 0
        speed_array_list = []
        angle_array_list = []
        min_shape = 100
        org_bbox = copy.deepcopy(bbox)
        self.bbox_dict = {}
        self.initial_id_count = None
        for bb in bbox:
            self.create_initial_id_to_bbox(bb)

        warped_prev_bbox_list = []
        for index, prev_bbox in enumerate(prev_bbox_list):
            warped_bbox = self.warp_prev_bb(prev_bbox,M_warp_list[index])
            warped_prev_bbox_list.append(warped_bbox)

        for key_id in self.bbox_dict.keys():
            item = self.bbox_dict[key_id]
            corresponding_box_list = item['bbox']
            box = corresponding_box_list[0]
            self.time_count = -1
            for time_index, warped_previous_bb in enumerate(warped_prev_bbox_list):
                corressponding_prev_bb = self.find_corressponding_prev_bb(warped_previous_bb, box)
                if len(corressponding_prev_bb) == 0:
                    continue
                box = corressponding_prev_bb
                corresponding_box_list.append(corressponding_prev_bb)
                self.time_count -= 1
                self.add_prev_bbox(key_id, corresponding_box_list)
        assert len(bbox) == len(self.bbox_dict.keys())
        colors = self.generate_colors()
        for index, key_id in enumerate(self.bbox_dict.keys()):
            item = self.bbox_dict[key_id]
            box = item['bbox']
            for time_index,bb in enumerate(box):
                try:
                    assert bb.shape == (4,2)
                except:
                    asd = 0
                mask = self.draw_bounding_box(bb)
                image[mask] = tuple((np.array(colors[index]) * (time_index / len(box))).astype(np.uint8))

        speed_list = []
        angle_list = []
        sign_list = []
        diff_list = []
        area_list = []
        ori_angle_list = []
        for index, key_id in enumerate(self.bbox_dict.keys()):
            item = self.bbox_dict[key_id]
            box = np.stack(item['bbox'], axis=0)
            centers_x_y = box.mean(-2)
            channel_speed_list = []
            channel_angle_list = []
            channel_sign_list = []
            channel_diff_list = []
            channel_current_ori_angle_list = []
            channel_area_list = []
            for index_x_y, cur_x_y in enumerate(centers_x_y):
                if index_x_y + 1== len(centers_x_y):
                    break
                prev_x_y = centers_x_y[index_x_y+1]
                prev_cor = np.linalg.norm(prev_x_y-ego_location)
                cur_cor = np.linalg.norm(cur_x_y-ego_location)
                ar= np.sign(cur_x_y-ego_location)
                array_element = self.get_area_names([ar])
                channel_area_list.append(array_element)

                current_ori_angle = self.get_angle(box[index_x_y])+90


                diff_cor = np.sign(cur_cor - prev_cor)
                if ((array_element[0]==3 or array_element[0]==2) and diff_cor == 1.0) or ((array_element[0]==4 or array_element[0]==1) and diff_cor == -1.0):
                    current_ori_angle += 180
                channel_current_ori_angle_list.append((current_ori_angle%360))#tugrul
                channel_diff_list.append(diff_cor)
                channel_sign_list.append(np.sign(np.sign(prev_x_y - centers_x_y).mean(0)))

                speed = (((cur_x_y[0] - prev_x_y[0]) ** 2 + (cur_x_y[1] - prev_x_y[1]) ** 2) ** 0.5 ) * 0.25
                channel_speed_list.append(speed)

            if len(channel_diff_list) != 0:
                diff_list.append(channel_diff_list[self.select_proper_index(channel_diff_list)])
                speed_list.append(channel_speed_list[self.select_proper_index(channel_speed_list)])
                ori_angle_list.append(channel_current_ori_angle_list[self.select_proper_index(channel_current_ori_angle_list)])
                area_list.append(channel_area_list[self.select_proper_index(channel_area_list)])
            else:
                diff_list.append(0)
                speed_list.append(0)
                area_list.append([0])
                ori_angle_list.append(0)

            asd = 0


        speed_array = np.array(speed_list).astype(np.float)
        sign_array = np.array(sign_list).astype(np.float)
        diff_array = np.array(diff_list).astype(np.float)
        area_array = np.array(area_list).astype(np.float)
        ori_angle_array = np.array(ori_angle_list).astype(np.float)

        print("speed_array:",speed_array)
        print("sign_array:",sign_array)
        print("diff_array:",diff_array)
        print("ori_angle_array:",ori_angle_array)
        color_name_list = self.get_name_of_colors(colors)
        print("color_name_list:",color_name_list)
        bbox_ori_angle = np.array(orientation_angle_list)
        render_dict = {'info':['color_name_list: '+str(color_name_list), 'diff_array: ' + str(diff_array),'area_array: '+str(area_array),'ori_angle_array:'+ str(ori_angle_array),'bbox_ori_angle: '+ str(bbox_ori_angle),'speed_array: '+str(speed_array)]}

        assert len(bbox) == len(ori_angle_array)
        assert len(bbox) == len(speed_array)
        return speed_array, np.radians(ori_angle_array), image, render_dict, color_name_list

    def update_render_dict(self,render_dict, control, close_points_count, mean_alt, mean_vel, is_there_obstacle):
        new_render_dict = {'info': render_dict['info']+['control: '+str(control), 'number of points: '+str(close_points_count), 'mean_alt: '+str(mean_alt), 'mean_vel: '+str(mean_vel), 'is_there_obstacle: ' + str(is_there_obstacle)]}
        return new_render_dict

    def get_angle(self,corners):
        # Calculate the centroid of the bounding box
        centroid = np.mean(corners, axis=0)

        # Center the points around the centroid
        centered_points = corners - centroid

        # Calculate the covariance matrix of the centered points
        cov_matrix = np.cov(centered_points, rowvar=False)

        # Compute the eigenvectors and eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # The eigenvector with the highest eigenvalue corresponds to the major axis
        major_axis_vector = eigenvectors[:, np.argmax(eigenvalues)]

        # Calculate the orientation angle
        orientation_angle = np.arctan2(major_axis_vector[1], major_axis_vector[0])
        orientation_angle = np.degrees(orientation_angle)

        return orientation_angle

    def get_name_of_colors(self,colors):
        cl_list = []
        for cl in colors:
            cl_list.append(self.closest_color(cl))
        return cl_list

    def closest_color(self, requested_color):
        min_colors = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_color[0]) ** 2
            gd = (g_c - requested_color[1]) ** 2
            bd = (b_c - requested_color[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        return min_colors[min(min_colors.keys())]


    def generate_colors(self):
        n = len(self.bbox_dict.keys())
        colors = []
        for i in range(n):
            # Evenly space the hue value and convert HSV to RGB
            hue = i / n
            rgb_color = colorsys.hsv_to_rgb(hue, 1, 1)

            # Scale the RGB values to 0-255
            rgb_color_scaled = tuple(int(c * 255) for c in rgb_color)
            colors.append(rgb_color_scaled)

        return colors

    def warp_prev_bb(self,prev_bbox,M_warp):
        warped_previous_bb = []
        for index, box in enumerate(prev_bbox):
            warped_previous_bb.append(self.plant_warp_pixel_location(box, M_warp))
        return warped_previous_bb

    def find_corressponding_prev_bb(self,warped_previous_bb,box):
        iou_list = []
        prev_box = []
        for _, prev_bbox in enumerate(warped_previous_bb):
            iou_score = self.calculate_iou(prev_bbox, box)
            iou_list.append(iou_score)
        if np.sum(iou_list) != 0:
            max_iou_index = np.argmax(iou_list)
            prev_box = warped_previous_bb[max_iou_index]

        return prev_box

    def plant_get_speed(self,bbox, prev_bbox, ego_motion, M_warp, compass, image):
        speed_array = np.zeros(len(bbox)).astype(np.float)
        if not isinstance(prev_bbox, type(None)):
            warped_previous_bb = []
            for index, box in enumerate(prev_bbox):
                warped_previous_bb.append(self.plant_warp_pixel_location(box,M_warp))
            mask_red = self.draw_bounding_box(warped_previous_bb)
            image[mask_red] = (255,0,0)
            mask_blue = self.draw_bounding_box(bbox)
            image[mask_blue] = (0,0,255)

        speed_array = np.zeros(len(bbox)).astype(np.float)
        angle_array = np.zeros(len(bbox)).astype(np.float)
        for index, box in enumerate(bbox):
            if not isinstance(self.previous_bb, type(None)):
                iou_list = []
                for _, prev_bbox in enumerate(warped_previous_bb):
                    iou_list.append(self.calculate_iou(prev_bbox, box))
                if np.sum(iou_list) != 0:
                    max_iou_index = np.argmax(iou_list)
                    prev_box = warped_previous_bb[max_iou_index]
                    prev_center_x, prev_center_y = prev_box.mean(0)[0], prev_box.mean(0)[1]
                    center_x, center_y = box.mean(0)[0], box.mean(0)[1]
                    angle = (self.calculate_angle(prev_center_x, prev_center_y, center_x, center_y))
                    #angle += math.degrees(compass)%360
                    speed = ((center_x-prev_center_x)**2+(center_y-prev_center_y)**2)**0.5
                    speed *= 1#4#pixel_per_meter
                    speed_array[index] = speed
                    angle_array[index] = angle
                else:
                    speed_array[index] = 7*4
        print("angle_array:",angle_array,"ego compass:",math.degrees(compass))
        self.previous_bb = bbox
        return speed_array, angle_array, image

    def plant_warp_pixel_location(self, past_pixel, M_warp):
        """
        Warp the past pixel location based on the ego motion.

        Parameters:
        past_pixel (tuple): The past pixel location of the other vehicle (x, y).
        ego_motion (numpy array): 1x6 vector [x, y, z, pitch, yaw, roll] of the ego vehicle.

        Returns:
        tuple: The warped pixel location.
        """
        """# Extract 2D translation and yaw rotation from ego_motion
        dx, dy, _, _, yaw, _ = ego_motion

        # Convert yaw to radians
        yaw_rad = np.radians(yaw)

        # Create a 2D rotation matrix
        rotation_matrix = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad), np.cos(yaw_rad)]
        ])"""

        # Apply rotation and then translation
        warped_pixel = []
        """for corner in past_pixel:
            cv2.transform(corner, M_warp)
            warped_corner = np.dot(rotation_matrix, corner) + np.array([dx, dy])
            warped_pixel.append(warped_corner)"""
        #warped_pixel = np.stack(warped_pixel,0)
        warped_pixel = cv2.transform(np.expand_dims(past_pixel, 1), M_warp)#cv2.warpAffine(img,M,(cols,rows))

        #print("warped_pixel:",warped_pixel,"past_pixel:",past_pixel,"yaw:",yaw,"dx, dy:",dx, dy)
        return warped_pixel.squeeze(1)

    def get_axis_aligned_bbox(self, points):
        """
        Convert a bounding box represented by four points into an axis-aligned bounding box.

        Parameters:
        points -- a tuple of four points, each point is a tuple (x, y)
        """
        x_coordinates, y_coordinates = zip(*points)
        return (min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates))

    def calculate_iou(self, box1, box2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters:
        box1 -- the first bounding box, a tuple of four points (x, y)
        box2 -- the second bounding box, a tuple of four points (x, y)
        """

        # Convert the boxes to axis-aligned bounding boxes
        box1 = self.get_axis_aligned_bbox(box1)
        box2 = self.get_axis_aligned_bbox(box2)

        # Determine the coordinates of the intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The area of intersection
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # The area of both AABBs
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # The area of the union
        union_area = box1_area + box2_area - intersection_area

        # IoU calculation
        iou = intersection_area / union_area

        return iou

    def draw_bounding_box(self, corners):
        # Create a blank image, white background

        mask = np.zeros((200, 200)).astype(np.uint8)

        # Draw the bounding box
        # Assuming corners are in the format [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        try:
            for cor in corners:
                cor = np.int0(cor)
                cv2.drawContours(mask, [cor], 0, (255), -1)
        except:
            cor = np.int0(corners)
            cv2.drawContours(mask, [cor], 0, (255), -1)

        return mask.astype(np.bool)

    def calculate_angle(self, x1, y1, x2, y2):
        #return math.degrees(math.atan2(y2 - y1, x2 - x1))
        return math.degrees(math.atan2(y2 - y1, x2 - x1))

    def get_area_names(self,area_arrays):
        area_name_list = []
        for area in area_arrays:
            if area[0] >= 0 and area[1] >= 0:
                area_name_list.append(1)
            elif area[0] >= 0 and area[1] <= 0:
                area_name_list.append(2)
            elif area[0] <= 0 and area[1] <= 0:
                area_name_list.append(3)
            elif area[0] <= 0 and area[1] >= 0:
                area_name_list.append(4)

        return np.stack(area_name_list,axis=0)
