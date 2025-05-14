import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from collections import deque
import carla

class Sensor_Fusion:
    def __init__(self,time_step=0.05):
        # Define the state transition matrix
        dt = time_step # time step
        self.dt = dt
        F = np.array([[1, dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, dt],
                      [0, 0, 0, 1]])

        # Define the measurement matrix for GPS (position only)
        H = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])

        # Define the initial state (position and velocity)
        x = np.array([0, 0, 0, 0])  # [x_position, x_velocity, y_position, y_velocity]

        # Define the initial covariance matrix
        P = np.eye(4) * 500

        # Define the process noise covariance matrix
        Q = Q_discrete_white_noise(dim=2, dt=1, var=0.1, block_size=2) #np.eye(4)

        # Define the measurement noise covariance matrix for GPS
        R = np.array([[5, 0],
                      [0, 5]])

        # Create a Kalman filter instance
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = F
        self.kf.H = H
        self.kf.x = x
        self.kf.P = P
        self.kf.Q = Q
        self.kf.R = R

        self.history_que = deque(maxlen=5)

        self.initial = True

    def __call__(self, input_data):
        sensor_location = self.get_gnss_data(input_data['gps'])

        self.history_que.append([np.array([sensor_location.x,sensor_location.y]), input_data['imu'][1]])

        """if self.initial:
            self.initial = False
            for i in range(40):
                self.calculate_kf()
        else:"""
        self.calculate_kf()

        x, y = self.kf.x[0], self.kf.x[2]

        return carla.Location(x=x, y=y, z=sensor_location.z)

    def calculate_kf(self):
        gps_data = self.history_que[-1][0]
        imu_data = self.history_que[-1][1]

        # Predict the state
        self.kf.predict()

        # Update with GPS measurement
        self.kf.update(gps_data)

        # Integrate IMU data to update velocity and position
        self.kf.x[1] += imu_data[0] * self.dt  # Update x_velocity with IMU acceleration
        self.kf.x[3] += imu_data[1] * self.dt  # Update y_velocity with IMU acceleration
        self.kf.x[0] += self.kf.x[1] * self.dt  # Update x_position with integrated velocity
        self.kf.x[2] += self.kf.x[3] * self.dt  # Update y_position with integrated velocity


    def get_gnss_data(self, raw_data):
        latitude, longitude, altitude = raw_data[1][0], raw_data[1][1], raw_data[1][2]
        lat_rad = (np.deg2rad(latitude) + np.pi) % (2 * np.pi) - np.pi
        lon_rad = (np.deg2rad(longitude) + np.pi) % (2 * np.pi) - np.pi
        R = 6378135  # Equatorial radius in meters
        x = R * np.sin(lon_rad) * np.cos(lat_rad)  # i0
        y = R * np.sin(-lat_rad)  # i0
        z = altitude
        return carla.Location(x=x,y=y,z=z)

