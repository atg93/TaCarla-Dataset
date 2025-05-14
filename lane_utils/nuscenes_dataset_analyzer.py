'''
This file is to preprocess nuScenes dataset to align unannotated data files. LIDAR is used as the reference sensor and
all other sensor readings are aligned to LIDAR's timestamps.
*** It can be improved based on our future needs ***
'''

import os
import copy
import json
import cv2
import numpy as np


def closest(lst, K):
    index = min(range(len(lst)), key=lambda i: abs(lst[i] - K))
    return lst[index], index

SWEEPS_PATH = '/datasets/nu/nuscenes/sweeps'

sensors = os.listdir(SWEEPS_PATH)
files = {}
scenes = []

sensor_ = sensors[0]
sensor_path = os.path.join(SWEEPS_PATH, sensor_)
sensor_data = os.listdir(sensor_path)
sensor_data.sort()
empty_dict = {}
for sensor in sensors:
    empty_dict[sensor] = []

for d in sensor_data:
    dList = d.split('__')
    if dList[0] not in files.keys():
        files[dList[0]] = copy.deepcopy(empty_dict)

for sensor in sensors:
    sensor_path = os.path.join(SWEEPS_PATH, sensor)
    sensor_data = os.listdir(sensor_path)
    sensor_data.sort()
    for d in sensor_data:
        dList = d.split('__')
        files[dList[0]][dList[1]].append(int(dList[2].split('.')[0]))
    sensor_data = []

aligned_scenes = {}
iter = 1
for sname in files.keys():
    #sname = 'n008-2018-05-21-11-06-59-0400'
    print('Scene {} processing is started...({} / {})'.format(sname, iter, len(files.keys())))
    scene = files[sname]
    reference_sensor = scene['LIDAR_TOP']
    aligned_scene = {}
    for sensor in sensors:
        lst = scene[sensor]
        temp = []
        for t in reference_sensor:
            time_ms, index = closest(lst, t)
            temp.append(time_ms)
        aligned_scene[sensor] = temp

    aligned_scenes[sname] = copy.deepcopy(aligned_scene)
    iter = iter + 1

with open('/workspace/ik22/track_data/aligned_scenes.json', 'w') as f:
    json.dump(aligned_scenes, f)

########################################################################################################################
########################################################################################################################
# PLOT CAMERAS TO CHECK WHETHER THEY ARE GOOD OR NOT
#
# sensors = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
# sname = 'n008-2018-05-21-11-06-59-0400'
# scene = aligned_scenes[sname]
# n_frame = len(scene['LIDAR_TOP'])
# SWEEPS_PATH = '/datasets/nu/nuscenes/sweeps'
# for idx in range(n_frame):
#     imgs = []
#     for sensor in sensors:
#         time_ms = scene[sensor][idx]
#         img_name = SWEEPS_PATH + '/' + sensor + '/' + sname + '__' + sensor + '__' + str(time_ms) + '.jpg'
#         img = cv2.imread(img_name)
#         resized_up = cv2.resize(img, (400, 225), interpolation=cv2.INTER_LINEAR)
#         imgs.append(resized_up)
#
#     h, w, c = imgs[0].shape
#     imgs1 = np.concatenate(imgs[:3], axis=1)
#     imgs2 = np.concatenate(imgs[3:], axis=1)
#     img_to_show = np.concatenate((imgs1, imgs2), axis=0)
#     #img_to_show = cv2.resize(imgs, dsize=(3*w, 2*h))
#     cv2.imshow("6 camera images", img_to_show)
#
#     print('frame {} is done'.format(idx))
#     ch = cv2.waitKey(1)
#     if ch == 27:
#         break
#
# cv2.destroyAllWindows()
########################################################################################################################
########################################################################################################################

with open('/datasets/nu/nuscenes/v1.0-trainval/ego_pose.json') as f:
    ego_pose = json.load(f)

with open('/datasets/nu/nuscenes/v1.0-trainval/calibrated_sensor.json') as f:
    calibrated_sensor = json.load(f)

with open('/datasets/nu/nuscenes/v1.0-trainval/sample_data.json') as f:
    sample_data = json.load(f)

sample_data_key_frames = []
for sd in sample_data:
    if sd['is_key_frame']:
        sample_data_key_frames.append(sd)

ego_tokens = []
calibrated_sensor_tokens = []
sensors = []
scenes = []
times = []
for sd in sample_data_key_frames:
    ego_tokens.append(sd['ego_pose_token'])
    calibrated_sensor_tokens.append(sd['calibrated_sensor_token'])
    tmp = sd['filename'].split('/')
    sensors.append(tmp[1])
    times.append(sd['timestamp'])
    tmp2 = tmp[2].split('__')
    scenes.append(tmp2[0])

print("SAMPLE DATA PROCESSING... DONE")
unique_scenes = np.unique(scenes)
unique_sensors = np.unique(sensors)

tokens = {}

for scene_id in range(unique_scenes.shape[0]):
    scene_name = unique_scenes[scene_id]
    filt = [i for i,t in enumerate(scenes) if t==scene_name]
    scene = [scenes[ii] for ii in filt]
    scene_times = [times[ii] for ii in filt]
    scene_ego_tokens = [ego_tokens[ii] for ii in filt]
    scene_calibrated_sensor_tokens = [calibrated_sensor_tokens[ii] for ii in filt]
    scene_sensors = [sensors[ii] for ii in filt]
    tokens[scene_name] = {}
    for sensor_id in range(unique_sensors.shape[0]):
        sensor_name = unique_sensors[sensor_id]
        tokens[scene_name][sensor_name] = {}
        inds = [i for i,t in enumerate(scene_sensors) if t==sensor_name]
        tokens[scene_name][sensor_name]['ego_pose_token'] = [scene_ego_tokens[ii] for ii in inds]
        tokens[scene_name][sensor_name]['calibrated_sensor_token'] = [scene_calibrated_sensor_tokens[ii] for ii in inds]
        tokens[scene_name][sensor_name]['timestamp'] = [scene_times[ii] for ii in inds]

print("SAMPLE DATA DICTIONARY PROCESSING... DONE")

aligned_scenes_new = {}
counter = 0
for scene_name in tokens.keys():
    if counter % 10 == 0:
        print("ITERATION:", counter)
    aligned_scenes_new[scene_name] = {}
    a_sc = aligned_scenes[scene_name]
    token = tokens[scene_name]
    for sensor_name in a_sc.keys():
        aligned_scenes_new[scene_name][sensor_name] = {}
        sensor_times = a_sc[sensor_name]
        token_sensor = token[sensor_name]

        aligned_scenes_new[scene_name][sensor_name]['timestamp'] = sensor_times
        ept = []
        cst = []
        for time in sensor_times:
            time_ms, index = closest(token_sensor['timestamp'], time)
            ept.append(token_sensor['ego_pose_token'][index])
            cst.append(token_sensor['calibrated_sensor_token'][index])

        aligned_scenes_new[scene_name][sensor_name]['ego_pose_token'] = ept
        aligned_scenes_new[scene_name][sensor_name]['calibrated_sensor_token'] = cst
    counter = counter + 1

with open('/workspace/ik22/track_data/aligned_scenes_new.json', 'w') as f:
    json.dump(aligned_scenes_new, f)
