from leaderboard.autoagents.traditional_agents_files.utils.wheel_controller import WheelController
from leaderboard.autoagents.traditional_agents_files.utils.keyboard_controller import KeyboardController

import pygame
import os

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import numpy as np
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class Manuel_Control():

    def __init__(self, gui_mode = True, countdown_seconds = 2,  speed_boost = 1.0,  manual_speed = 0.8,  wheel_control =
    True,  force_feedback = True,  wheel_turning_threshold = 0.05):
        self.wheel_control = wheel_control
        print("pygame initialized")
        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((1600, 1200), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Autonomous Driving Demo")

        self.mode_counter = countdown_seconds * 20

        if wheel_control: #initialize steering wheel
            self.controller = WheelController()
            self.controller.auto_center()
        else:
            self.controller = KeyboardController(manual_speed)

        self._prev_timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp.frame
        self.button_press = False


    def __call__(self, input_data, ego_vel, bev_data,file_name_without_extension):
        self.run_interface(input_data, ego_vel, bev_data, file_name_without_extension)
        self.get_control()
        control = self.controller.manual_control
        if self.wheel_control:
            self.button_press = self.controller.get_reverse_button_press()
        return control, self.button_press

    def run_interface(self, input_data, ego_vel, bev_data, file_name_without_extension):
        """
        Run the GUI
        """
        #font_path = "CotrellCFExtraBold-ExtraBold.ttf"
        font = ImageFont.truetype("arial.ttf", 50)
        # process sensor data
        image_center = input_data['front'][1][:, :, -2::-1]
        #image_center = cv2.resize(image_center, dsize=(800, 600), interpolation=cv2.INTER_CUBIC)
        if not bev_data==None:
            image_center[-200::, -200::] = bev_data

        if self.controller.mode_changed:
            if self.controller.mode_manual:
                text = "Switching to autonomous mode in" + str(self.mode_counter // 20) + "seconds !!!"
            else:
                text = "Switching to manuel mode in" + str(self.mode_counter // 20) + "seconds !!!"
            self.mode_counter -= 1
            if self.mode_counter == 0:
                self.controller.mode_manual = True if self.controller.mode_manual is False else False
                self.controller.mode_changed = False
                self.mode_counter = self.variables["countdown_seconds"] * 20
                if self.controller.mode_manual:
                    self._om_handler._obs_managers['birdview'].compute_nuplan = False
                else:
                    self._om_handler._obs_managers['birdview'].compute_nuplan = True
        else:
            if self.controller.mode_manual:
                text = "Driving in manuel mode !!!"
            else:
                text = "Driving in autonomous mode !!!"
            text = str(int(ego_vel))
        img_PIL = Image.fromarray(image_center)
        ImageDraw.Draw(img_PIL).text((50, 50), file_name_without_extension, font=font, fill=(255, 0, 0))
        ImageDraw.Draw(img_PIL).text((50, 100), text, font=font, fill=(255, 0, 0))
        image_center = np.array(img_PIL)

        # display image
        self._surface = pygame.surfarray.make_surface(image_center.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def get_control(self):
        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp.frame
        if self.wheel_control:
            self.controller.parse_vehicle_wheel()
        else:
            self.controller.parse_vehicle_keys(pygame.key.get_pressed(), (timestamp - self._prev_timestamp) * 1000)

        self._prev_timestamp = timestamp
