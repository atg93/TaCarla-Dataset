# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

from typing import Optional


def category_to_detection_name(category_name: str, detection_mapping: dict = None) -> Optional[str]:
    """
    Default label mapping from nuScenes to nuScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic nuScenes class.
    :param detection_mapping: Detection class mapping.
    :return: nuScenes detection class.
    """
    if detection_mapping is None:
        detection_mapping = {
            'movable_object.barrier': 'barrier',
            'vehicle.bicycle': 'bicycle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.car': 'car',
            'vehicle.construction': 'construction_vehicle',
            'vehicle.motorcycle': 'motorcycle',
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'movable_object.trafficcone': 'traffic_cone',
            'vehicle.trailer': 'trailer',
            'vehicle.truck': 'truck'
        }

    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        return None
