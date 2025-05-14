#!/usr/bin/env python3

import argparse
import json
import logging

import carla
import cv2
import math
from custom_client.simulation_module import SetTown, SetWeather
# from ipm.ipm import IPM
from tairvision.utils import IPM
import numpy as np
import re
import os
import os.path as osp
import traceback

try:
    from custom_client.simulation_module import Simulation
    from custom_client.simulation_module import ReadSimulationParameters
    from custom_client.simulation_module import CameraSensor2CvImage
    from custom_client.simulation_module import LaneDetector
    from custom_client.simulation_module.carla_module.HelperFunctions import GetCameraProjectionMatrix
    from custom_client.simulation_module.lane_module.LaneFunctions import DrawLineArrayOnImage, GetCulaneColors, LaneMaskImage2MaskData
except ModuleNotFoundError:
    logging.error("Simulation Module not found. Set workspace parent directory.")
    exit(1)


class LaneDataIterative:

    def __init__(self, sim_parameters, next_point_distance, lane_change_freq, random_teleport_freq,
                 weather_change_freq, duration, image_path, annotation_path):
        # sim_parameters = ReadSimulationParameters(arg.config, arg.sim)

        # loggine level
        DEBUG = False
        # parameter for lane detector
        max_distance = 30
        point_number = 40
        self.duration = duration
        junction_mode = True

        self.town = sim_parameters['town']
        self.image_path = osp.join(image_path, self.town)
        self.annotation_path = annotation_path

        if not osp.exists(self.image_path):
            os.makedirs(self.image_path)

        if not osp.exists(self.annotation_path):
            os.makedirs(self.annotation_path)

        self.simulation = Simulation(sim_parameters, debug=DEBUG, multiple_simulation=True)
        self.simulation.set_parameters(next_point_distance=next_point_distance)
        self.lane_detector = LaneDetector(max_distance, point_number, junction_mode)
        self.deviation_counter = 0
        self.counter = 0
        self.ipm = None

        self.lane_change_freq = lane_change_freq
        self.random_teleport_freq = random_teleport_freq
        self.weather_change_freq = weather_change_freq
        self.next_point_distance = next_point_distance

        self.weather_time = [
        'HardRainNoon',
        'HardRainSunset',
        'HardRainNight',
         'ClearNight',
         'ClearNoon',
         'ClearSunset',
         'CloudyNight',
         'CloudyNoon',
         'CloudySunset',
         'SoftRainNight',
         'SoftRainNoon',
         'SoftRainSunset',
         'WetCloudyNight',
         'WetCloudyNoon',
         'WetCloudySunset',
         'WetNight',
         'WetNoon',
         'WetSunset',
         'MidRainSunset',
         'MidRainyNight',
         'MidRainyNoon',
        ]

        self.json_dict = dict(frames=[], config=dict())

        # self.game_loop()

    def __enter__(self):
        return self

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     self.simulation.stop_simulation()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self

    def game_iter(self):
        weather_changed = False
        weather = self.weather_time[(self.counter//self.weather_change_freq) % len(self.weather_time)]
        if self.counter % self.weather_change_freq == 0:
            self.simulation.set_weather(weather)
            weather_changed = True
            if 'Night' in weather:
                self.simulation.ego_vehicle.SetLights(low_beam=True, position=True)
                self.simulation.NPC.SetLights(low_beam=True, position=True)
            else:
                self.simulation.ego_vehicle.SetLights(low_beam=False, position=False)
                self.simulation.NPC.SetLights(low_beam=False, position=False)

        snapshot, temp_sensor_data = self.simulation.step()
        camera_image_data = temp_sensor_data[0]

        camera_matrix = GetCameraProjectionMatrix(temp_sensor_data[0])
        camera_img = CameraSensor2CvImage(camera_image_data)

        if self.ipm is None:
            self.ipm = IPM(camera_matrix[0,0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2],
                           x=0.8, y=0.0, z=1.3, yaw=0, pitch=0, roll=0, img_shape=camera_img.shape)


        lane_points = self.lane_detector.CalculateLanes(self.simulation.Map, camera_image_data)
        masked_image, masked_data = self.lane_detector.GetMaskImageLine()

        # if self.counter % 20 == 0:
        #     self.simulation.set_building_visibility(0.5)
        #     # self.simulation.set_vegetation_visibility(0.5)

        if weather_changed:
            self.simulation.ego_teleport_safe_location()
        else:
            if self.counter % self.lane_change_freq == 0:
                self.simulation.ego_lane_change()
            elif self.counter % self.random_teleport_freq == 0:
                self.simulation.ego_send_random_point()
            else:
                success = self.simulation.ego_teleport_safe_location()
                if success != 1:
                    self.simulation.ego_send_random_point()


        lane_points = self._mask_lane_point(lane_points, camera_img.shape)

        ipm_lane_points = [None for _ in lane_points]
        for i, points in enumerate(lane_points):
            if points is not None:
                points_np = np.array(points)
                proj_points = cv2.perspectiveTransform(points_np[:, None, :].astype('float'), self.ipm.IPM)
                ipm_lane_points[i] = proj_points[:, 0, :].round().astype('int')

        warped_img = self.ipm.front_2_bev(camera_img)
        # ipm_mask_image = np.zeros_like(warped_img).astype('uint8')
        warped_img = DrawLineArrayOnImage(warped_img, ipm_lane_points, GetCulaneColors(), sort_y=False)
        # warped_img = LaneMaskImage2MaskData(ipm_mask_image)

        warped_img = cv2.rotate(warped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # ipm_mask_data = cv2.rotate(ipm_mask_data, cv2.ROTATE_90_COUNTERCLOCKWISE)

        visualization_img = camera_img.copy()

        visualization_img[masked_image != 0] = masked_image[masked_image != 0]

        weather_codes = [a for a in re.split(r'([A-Z][a-z]*)', weather) if a]
        attributes = dict(timeofday=weather_codes[-1].lower(),
                          weather=" ".join(weather_codes[:-1]).lower())
        camera_params = dict(fx=camera_matrix[0,0], fy=camera_matrix[1,1],
                             u0=camera_matrix[0,2], v0=camera_matrix[1,2],
                              x=0.8, y=0.0, z=1.3, yaw=0, pitch=0, roll=0)

        frame_dict = dict(frame_index=self.counter,
                          name="{:08d}_img_front.jpg".format(self.counter),
                          camera_params=camera_params,
                          lane_points=lane_points,
                          attributes=attributes,
                          town=self.town)

        self.counter += 1


        return camera_img, frame_dict, warped_img, visualization_img

    @staticmethod
    def _mask_lane_point(lane_points, img_shape):
        for i, points in enumerate(lane_points):
            if points is not None:
                lane_points[i] = np.stack(points)
                mask_x = np.logical_and(lane_points[i][:, 0] > 0, lane_points[i][:, 0] < img_shape[1])
                mask_y = np.logical_and(lane_points[i][:, 1] > 0, lane_points[i][:, 1] < img_shape[0])
                mask = np.logical_and(mask_x, mask_y)
                lane_points[i] = lane_points[i][mask]
                if lane_points[i].shape[0] != 0:
                    lane_points[i] = lane_points[i].tolist()
                else:
                    lane_points[i] = None
        return lane_points

    def game_loop(self):
        try:
            counter = 0
            while counter < self.duration:
                camera_img, frame_dict, warped_img, vis_img = self.game_iter()
                self.json_dict['frames'].append(frame_dict)
                self.json_dict['config']['image_size'] = dict(width=camera_img.shape[1], height=camera_img.shape[0])
                cv2.imwrite(osp.join(self.image_path, frame_dict['name']), camera_img)
                cv2.imwrite(osp.join(self.image_path, frame_dict['name'].rsplit('.', 1)[0] + '_bev.jpg'), warped_img)
                cv2.imwrite(osp.join(self.image_path, frame_dict['name'].rsplit('.', 1)[0] + '_vis.jpg'), vis_img)

                # cv2.imshow("mask", masked_image)
                cv2.imshow("camera", camera_img)
                cv2.imshow("warped", warped_img)
                cv2.imshow("visualization", vis_img)

                cv2.waitKey(1)
                counter += 1

                if counter % 100 == 0:
                    logging.info(" --- {} frame extraction %{:.2f} has been completed ---".format(self.town, (counter/self.duration)*100))

        except Exception as e:
            traceback.print_exc()
        finally:
            with open(osp.join(self.annotation_path, self.town + '.json'), 'w') as fp:
                json.dump(self.json_dict, fp)
            self.simulation.stop_simulation()


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(
        description=__doc__)

    argument_parser.add_argument(
        '--config',
        default="simulation_config.json",
        help='config file')

    argument_parser.add_argument(
        '--sim',
        default="base_simulation",
        help='simulation selection')

    argument_parser.add_argument(
        '--town',
        default="Town01"
    )

    argument_parser.add_argument(
        '--weather',
        default="ClearNight"
    )

    args = argument_parser.parse_args()

    # LaneDataQueue(args)
