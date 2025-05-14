from typing import Tuple, List, Optional, Union, Dict
import numpy as np
import cv2
import matplotlib


class Instance:
    def __init__(self, info_dict: Dict[str, all]):

        self.box: np.ndarray = info_dict.get('box')
        self.score: float = info_dict.get('score')
        self.keypoint: Optional[np.ndarray] = info_dict.get('keypoint', None)

        self.box_history: List[np.ndarray] = []
        self.keypoint_history: List[np.ndarray] = []

        self.instanceID: int = None
        self.classID: Optional[int] = info_dict.get('classID', None)

        self.edges = [
            (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
            (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
            (12, 14), (14, 16), (5, 6)
        ]

        self.update(info_dict)

    def update(self, info_dict: Dict[str, all]):
        self.box_history = info_dict.get('box_history', [])
        self.keypoint_history = info_dict.get('keypoint_history', [])
        self.box_history.append(self.box)

        self.instanceID = info_dict.get('instanceID')

        if self.keypoint is not None:
            self.keypoint_history.append(self.keypoint)

    def create_instance_dict(self):
        instance_dict = {'instanceID': self.instanceID,
                         'box_history': self.box_history,
                         'keypoint_history': self.keypoint_history}

        return instance_dict

    def visualize(self, frame):
        x1 = int(self.box[0])
        x2 = int(self.box[2])
        y1 = int(self.box[1])
        y2 = int(self.box[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        image_h, _, _ = frame.shape
        # cv2.putText(frame,
        #             f"{self.score:0.2f}",
        #             (x1, y1 - 13),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             1e-3 * image_h,
        #             (0, 255, 0), 2)

        cv2.putText(frame,
                    f"{self.instanceID}",
                    (x1, y1 - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image_h,
                    (0, 255, 0), 2)

        if self.keypoint is not None:
            keypoints = self.keypoint
            for point_index in range(17):
                cv2.circle(frame, (int(keypoints[point_index, 0]), int(keypoints[point_index, 1])),
                           3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            for ie, e in enumerate(self.edges):
                # get different colors for the edges
                rgb = matplotlib.colors.hsv_to_rgb([
                    ie / float(len(self.edges)), 1.0, 1.0
                ])
                rgb = rgb * 255
                # join the keypoint pairs to draw the skeletal structure
                cv2.line(frame, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                         (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                         tuple(rgb), 2, lineType=cv2.LINE_AA)
