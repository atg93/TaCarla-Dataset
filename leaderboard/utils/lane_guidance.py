import carla
import numpy as np
import cv2
import math


class Lane_Guidance():
    def __init__(self,lane_class):
        self.lane_class = lane_class
        self.real_height = 586
        self.real_width = 1034


    def draw_box(self, stop_masks, center_x, center_y, box_width=10, box_height=10):
        # Calculate top-left and bottom-right points from the center
        start_x = center_x - box_width // 2
        start_y = center_y - box_height // 2
        end_x = center_x + box_width // 2
        end_y = center_y + box_height // 2

        # Step 2: Draw the rectangle
        cv2.rectangle(stop_masks, (start_x, start_y), (end_x, end_y), (255, 0, 0), -1)

        return stop_masks


    def crop_array_center(self, original_array, new_height, new_width):
        original_height, original_width = original_array.shape

        # Calculate the starting points for the crop
        start_row = (original_height - new_height) // 2
        start_col = (original_width - new_width) // 2

        # Calculate the ending points for the crop
        end_row = start_row + new_height
        end_col = start_col + new_width

        # Crop the array
        cropped_array = original_array[start_row:end_row, start_col:end_col]

        return cropped_array


    def __call__(self, input_data, _global_plan_world_coord, vehicle):
        guidance_masks = np.zeros([2000, 2000], dtype=np.uint8)
        lane_guidance = self.lane_class.get_lane_guidance()
        current_wp, _ = self.lane_class.get_related_wp()
        lane_list = []
        for ln, _ in lane_guidance:
            yaw = self.lane_class.convert_heading_to_yaw(math.degrees(input_data['imu'][1][-1]))
            #calculated_transform_vector = carla.Transform(rotation=carla.Rotation(pitch=ln.rotation.pitch,roll=ln.rotation.roll,yaw=yaw),location=current_wp.location)
            calculated_transform_vector = vehicle.get_transform() #carla.Transform(rotation=carla.Rotation(pitch=ln.rotation.pitch,roll=ln.rotation.roll,yaw=yaw),location=current_wp.location)
            new_ln_location = self.lane_class.world_to_vehicle(ln.location, calculated_transform_vector)
            lane_list.append([new_ln_location.x, new_ln_location.y, new_ln_location.z])
        projected_route_list = self.lane_class.project(np.array(lane_list))

        lane_image = np.zeros([200, 200], dtype=np.uint8)
        for lane_wp in lane_list:
            lane_wp = np.array(lane_wp[0:-1]).astype(np.uint8)
            lane_wp[0] = lane_wp[0] #* (-1)
            lane_wp = 100 - (lane_wp * 4)
            cv2.circle(lane_image, tuple(lane_wp), 3, (255), -1)
            #cv2.imwrite('lane_image.png',lane_image)

        for index, gm in enumerate(projected_route_list):
            try:
                y = int(gm[0])
                x = int(gm[1])

                y_1 = int(projected_route_list[index + 1][0])
                x_1 = int(projected_route_list[index + 1][1])

                cv2.line(guidance_masks, (x, y), (x_1, y_1), (255, 255, 255), 30)
            except:
                pass

        guidance_masks = self.crop_array_center(guidance_masks, self.real_height, self.real_width)

        return guidance_masks, lane_list, lane_image