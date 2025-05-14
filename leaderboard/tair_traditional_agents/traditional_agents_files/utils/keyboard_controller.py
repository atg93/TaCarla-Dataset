import pygame
import carla
from pygame.locals import K_DOWN
from pygame.locals import K_LEFT
from pygame.locals import K_RIGHT
from pygame.locals import K_SPACE
from pygame.locals import K_UP
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_s
from pygame.locals import K_w
from pygame.locals import K_q


class KeyboardController:
    def __init__(self, manual_speed):
        self._steer_cache = 0.0
        self.manual_control = carla.VehicleControl()
        self.mode_manual = True
        self.manual_speed = manual_speed
        self.mode_changed = False
        self.throttle_toggle = False
        self.steer_toggle = False

    def parse_vehicle_keys(self, keys, miliseconds):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYUP:
                if event.key == K_q:
                    self.manual_control.gear = 1 if self.manual_control.reverse else -1
                    self.manual_control.reverse = self.manual_control.gear < 0

        if keys[K_SPACE]:
            self.mode_changed = True

        if self.mode_manual:

            if (keys[K_UP] or keys[K_w]) and self.throttle_toggle:
                self.manual_control.throttle = self.manual_speed
                self.throttle_toggle = False
            else:
                self.manual_control.throttle = 0.0
                self.throttle_toggle = True

            steer_increment = 3e-4 * miliseconds
            if keys[K_LEFT] or keys[K_a]:
                self._steer_cache -= steer_increment
            elif keys[K_RIGHT] or keys[K_d]:
                self._steer_cache += steer_increment
            else:
                self._steer_cache = 0.0

            self.steer_cache = min(0.95, max(-0.95, self._steer_cache))
            self.manual_control.steer = round(self._steer_cache, 1)
            self.manual_control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
            # self._control.hand_brake = keys[K_SPACE]
