import pygame
import time
from leaderboard.autoagents.traditional_agents_files.utils.force_feedback import ForceFeedback

import math
import carla


class WheelController:
    def __init__(self):
        time.sleep(2)
        pygame.joystick.init()
        time.sleep(2)
        # joystick_count = pygame.joystick.get_count()
        # if joystick_count > 1:
        #    raise ValueError("Please Connect Just One Joystick")
        device_number = 0
        self._joystick = pygame.joystick.Joystick(device_number)
        self._joystick.init()
        self._steer_idx = 0
        self._throttle_idx = 2
        self._brake_idx = 3
        self._reverse_idx = 5
        self._handbrake_idx = 4

        self.ForceFeedback = ForceFeedback(device_number)
        self.manual_control = carla.VehicleControl()
        self.mode_manual = True
        self.mode_changed = False
        self.reverse_button_press = False#tugrul

    def auto_center(self):
        self.ForceFeedback.autocenter_wheel()

    def get_reverse_button_press(self):
        return self.reverse_button_press

    def parse_vehicle_wheel(self):
        # pygame.event.pump()
        self.reverse_button_press = False
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.JOYBUTTONDOWN:
                print("event.button:",event.button)

                if event.button == self._reverse_idx:
                    self.reverse_button_press = True
                    pass
                    #self.manual_control.gear = 1 if self.manual_control.reverse else -1
                    #self.manual_control.reverse = self.manual_control.gear < 0

                """elif event.button == 2:
                    self.mode_changed = True"""

        if self.mode_manual:

            numAxes = self._joystick.get_numaxes()
            jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
            # print("inputs",jsInputs)
            jsButtons = [float(self._joystick.get_button(i)) for i in
                         range(self._joystick.get_numbuttons())]

            # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
            # For the steering, it seems fine as it is
            K1 = 1.0  # 0.55
            steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

            K2 = 1.6  # 1.6
            throttleCmd = K2 + (1.05 * math.log10(
                -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
            if throttleCmd <= 0:
                throttleCmd = 0
            elif throttleCmd > 1:
                throttleCmd = 1

            brakeCmd = 1.6 + (2.05 * math.log10(
                -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
            if brakeCmd <= 0:
                brakeCmd = 0
            elif brakeCmd > 1:
                brakeCmd = 1

            self.manual_control.steer = steerCmd
            self.manual_control.brake = brakeCmd
            self.manual_control.throttle = throttleCmd

            # toggle = jsButtons[self._reverse_idx]

            self.manual_control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def feedback_effect(self, control, thresh):
        self.ForceFeedback.done_right()
        self.ForceFeedback.done_left()

        turn = "center"  # represents which direction to force
        current_val = float(self._joystick.get_axis(0))
        difference = current_val - control.steer
        if abs(difference) > thresh:
            if difference < 0:
                turn = "right"
            elif difference > 0:
                turn = "left"

        if turn == "left":  # turn wheel left
            self.ForceFeedback.erase_effect(False)
            self.ForceFeedback.left_turn_wheel(abs(difference))

        elif turn == "right":  # turn wheel right
            self.ForceFeedback.erase_effect(True)
            self.ForceFeedback.right_turn_wheel(abs(difference))

        else:  # center
            self.ForceFeedback.erase_effect(True)
            self.ForceFeedback.erase_effect(False)
