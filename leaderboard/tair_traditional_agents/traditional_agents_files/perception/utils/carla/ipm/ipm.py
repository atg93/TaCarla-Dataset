import os
import sys
import yaml
import numpy as np
import cv2
import argparse
from tqdm import tqdm


class Camera:

  K = np.zeros([3, 3])
  R = np.zeros([3, 3])
  t = np.zeros([3, 1])
  P = np.zeros([3, 4])

  def setK(self, fx, fy, px, py):
    self.K[0, 0] = fx
    self.K[1, 1] = fy
    self.K[0, 2] = px
    self.K[1, 2] = py
    self.K[2, 2] = 1.0

  def setR(self, y, p, r):

    Rz = np.array([[np.cos(-y), -np.sin(-y), 0.0], [np.sin(-y), np.cos(-y), 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[np.cos(-p), 0.0, np.sin(-p)], [0.0, 1.0, 0.0], [-np.sin(-p), 0.0, np.cos(-p)]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(-r), -np.sin(-r)], [0.0, np.sin(-r), np.cos(-r)]])
    Rs = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]) # switch axes (x = -y, y = -z, z = x)
    self.R = Rs.dot(Rz.dot(Ry.dot(Rx)))

  def setT(self, XCam, YCam, ZCam):
    X = np.array([XCam, YCam, ZCam])
    self.t = -self.R.dot(X)

  def updateP(self):
    Rt = np.zeros([3, 4])
    Rt[0:3, 0:3] = self.R
    Rt[0:3, 3] = self.t
    self.P = self.K.dot(Rt)

  def __init__(self, config):
    self.setK(config["fx"], config["fy"], config["px"], config["py"])
    self.setR(np.deg2rad(config["yaw"]), np.deg2rad(config["pitch"]), np.deg2rad(config["roll"]))
    self.setT(config["XCam"], config["YCam"], config["ZCam"])
    self.updateP()

class IPM:
  def __init__(self, fx, fy, u0, v0, x, y, z, yaw, pitch, roll, img_shape):

    self.img_shape = img_shape
    camera_config = dict(fx=fx, fy=fy, px=u0, py=v0, XCam=x, YCam=y, ZCam=z, yaw=yaw, pitch=pitch, roll=roll)

    drone_config = dict(fx=682.578, fy=682.578, px=482.0, py=302.0, XCam=0.0, YCam=0.0, ZCam=18.0)
    cam = Camera(camera_config)

    outputRes = (int(2 * drone_config["py"]), int(2 * drone_config["px"]))
    self.outputRes = outputRes
    dx = outputRes[1] / drone_config["fx"] * drone_config["ZCam"]
    dy = outputRes[0] / drone_config["fy"] * drone_config["ZCam"]
    pxPerM = (outputRes[0] / dy, outputRes[1] / dx)

    # setup mapping from street/top-image plane to world coords
    # shift = (outputRes[0] / 2.0, outputRes[1] / 2.0)
    shift = (outputRes[0] / 2.0, -120.0)

    shift = shift[0] + drone_config["YCam"] * pxPerM[0], shift[1] - drone_config["XCam"] * pxPerM[1]
    M = np.array([[1.0 / pxPerM[1], 0.0, -shift[1] / pxPerM[1]],
                  [0.0, -1.0 / pxPerM[0], shift[0] / pxPerM[0]], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # find IPM as inverse of P*M
    self.IPM = np.linalg.inv(cam.P.dot(M))
    self.inv_IPM = cam.P.dot(M)

  def warp_img(self, image):
      warped1 = cv2.warpPerspective(image, self.IPM, (self.outputRes[1], self.outputRes[0]), flags=cv2.INTER_LINEAR)
      return warped1
  def bev_2_front(self, bev):
      front = cv2.warpPerspective(bev, self.inv_IPM, (self.img_shape[1], self.img_shape[0]), flags=cv2.INTER_LINEAR)
      return front
