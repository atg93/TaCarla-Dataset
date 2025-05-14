import numpy as np

BDD100K_DRIVABLE = np.array([
    [1, 120, 1],
    [1, 50, 120],
    [0, 0, 0],
    [120, 50, 1]
])

BDD100K_LANE_CONVERT = np.array([
    [0, 0, 0],
    [255, 1, 1],
    [1, 255, 1],
    [1, 1, 1]
])

BDD100K_LANE_NO_CONVERT = np.array([
    [0, 0, 0],  # 0 Bakcgorund
    [255, 255, 255],  # 1 Single White
    [51, 255, 1],  # 2 Single White Dashed
    [1, 255, 255],  # 3 Double White
    [1, 102, 255],  # 4 Double White Dashed
    [50, 1, 255],  # 5 Double Yellow
    [204, 1, 255],  # 6 Single Yellow
    [255, 1, 152],  # 7 Double Yellow Dashed
    [200, 211, 1],  # 8 Single Yellow Dashed
    [1, 1, 1]])

CULANE = np.array([
    [0, 0, 0],
    [180, 220, 30],
    [200, 100, 1],
    [1, 100, 200],
    [30, 200, 180]
])

BDD100K_SINGLE_LANE = np.array([
    [0, 0, 0],
    [255, 255, 255],
])

OPENLANE_LANE_TYPE = np.array([[  0,   0,   0],
        [147, 112, 219],
        [ 72, 209, 204],
        [186,  85, 211],
        [135, 206, 250],
        [255, 105, 180],
        [100, 149, 237],
        [154, 205,  50],
        [ 30, 144, 255],
        [250, 128, 114],
        [240, 128, 128],
        [255, 127,  80],
        [ 32, 178, 170],
        [255, 215,   0],
        [255, 215,   0],
        [  0,   0,   0]])


CITYSCAPES = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 1, 1],
    [1, 1, 142],
    [1, 1, 70],
    [1, 60, 100],
    [1, 80, 100],
    [1, 1, 230],
    [119, 11, 32],
    [0, 0, 0]
])

BDD100K_SEMANTIC_SEGMENTATION = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 1],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 1, 1],
    [1, 1, 142],
    [1, 1, 70],
    [1, 60, 100],
    [1, 80, 100],
    [1, 1, 230],
    [119, 11, 32],
])

MAPILLARY = np.array([
    [0, 0, 0],  # 0
    [0, 0, 0],  # 1
    [0, 0, 0],  # 2
    [0, 0, 0],  # 3
    [0, 0, 0],  # 4
    [0, 0, 0],  # 5
    [0, 0, 0],  # 6
    [0, 0, 0],  # 7
    [0, 0, 0],  # 8
    [0, 0, 0],  # 9
    [0, 0, 0],  # 10
    [0, 0, 0],  # 11
    [0, 0, 0],  # 12
    [0, 0, 0],  # 13
    [0, 0, 0],  # 14
    [0, 0, 0],  # 15
    [0, 0, 0],  # 16
    [0, 0, 0],  # 17
    [0, 0, 0],  # 18
    [0, 0, 0],  # 19
    [0, 0, 0],  # 20
    [0, 0, 0],  # 21
    [0, 0, 0],  # 22
    [0, 0, 0],  # 23
    [0, 0, 0],  # 24
    [0, 0, 0],  # 25
    [0, 0, 0],  # 26 Building 70 70 70
    [0, 0, 0],  # 27
    [0, 0, 0],  # 28
    [220, 20, 60],  # 29
    [255, 1, 1],  # 30
    [255, 1, 100],  # 31
    [255, 1, 200],  # 32
    [0, 0, 0],  # 33
    [0, 0, 0],  # 34
    [250, 170, 29],  # 35
    [250, 170, 26],  # 36
    [250, 170, 25],  # 37
    [250, 170, 24],  # 38
    [250, 170, 22],  # 39
    [250, 170, 21],  # 40
    [250, 170, 20],  # 41
    [255, 255, 255],  # 42
    [250, 170, 19],  # 43
    [250, 170, 18],  # 44
    [250, 170, 12],  # 45
    [250, 170, 11],  # 46
    [255, 255, 255],  # 47
    [255, 255, 255],  # 48
    [250, 170, 16],  # 49
    [250, 170, 15],  # 50
    [250, 170, 15],  # 51
    [0, 0, 0],  # 52
    [0, 0, 0],  # 53
    [0, 0, 0],  # 54
    [0, 0, 0],  # 55
    [0, 0, 0],  # 56
    [0, 0, 0],  # 57
    [0, 0, 0],  # 58
    [0, 0, 0],  # 59
    [0, 0, 0],  # 60
    [0, 0, 0],  # 61 Vegetation 107, 142, 35
    [0, 0, 0],  # 62
    [0, 0, 0],  # 63
    [0, 0, 0],  # 64
    [0, 0, 0],  # 65
    [0, 0, 0],  # 66
    [0, 0, 0],  # 67
    [0, 0, 0],  # 68
    [0, 0, 0],  # 69
    [0, 0, 0],  # 70
    [0, 0, 0],  # 71
    [0, 0, 0],  # 72
    [0, 0, 0],  # 73
    [0, 0, 0],  # 74
    [0, 0, 0],  # 75
    [0, 0, 0],  # 76
    [0, 0, 0],  # 77
    [0, 0, 0],  # 78
    [0, 0, 0],  # 79
    [0, 0, 0],  # 80
    [0, 0, 0],  # 81
    [0, 0, 0],  # 82
    [0, 0, 0],  # 83
    [0, 0, 0],  # 84
    [250, 170, 30],  # 85
    [250, 170, 30],  # 86
    [250, 170, 30],  # 87
    [250, 170, 30],  # 88
    [250, 170, 30],  # 89
    [250, 170, 30],  # 90
    [192, 192, 192],  # 91
    [192, 192, 192],  # 92
    [220, 220, 1],  # 93
    [220, 220, 1],  # 94
    [1, 1, 196],  # 95
    [192, 192, 192],  # 96
    [220, 220, 0],  # 97
    [0, 0, 0],  # 98
    [119, 11, 32],  # 99
    [0, 0, 0],  # 100
    [1, 60, 100],  # 101
    [1, 1, 142],  # 102
    [1, 1, 90],  # 103
    [1, 1, 230],  # 104
    [0, 80, 100],  # 105
    [128, 64, 64],  # 106
    [0, 0, 110],  # 107
    [1, 1, 70],  # 108
    [0, 0, 0],  # 109
    [0, 0, 0],  # 110
    [0, 0, 0],  # 111
    [0, 0, 0],  # 112
    [0, 0, 0],  # 113
    [0, 0, 0],  # 114
    [0, 0, 0]
])  # 115

MAPILLARY_ORIGINAL = np.array([
    [165, 42, 42],
    [1, 192, 1],
    [250, 170, 32],
    [196, 196, 196],
    [190, 153, 153],
    [180, 165, 180],
    [90, 120, 150],
    [250, 170, 33],
    [250, 170, 34],
    [128, 128, 128],
    [250, 170, 35],
    [102, 102, 156],
    [128, 64, 255],
    [140, 140, 200],
    [170, 170, 170],
    [250, 170, 36],
    [250, 170, 160],
    [250, 170, 37],
    [96, 96, 96],
    [230, 150, 140],
    [128, 64, 128],
    [110, 110, 110],
    [110, 110, 110],
    [244, 35, 232],
    [128, 196, 128],
    [150, 100, 100],
    [70, 70, 70],
    [150, 150, 150],
    [150, 120, 90],
    [220, 20, 60],
    [255, 1, 1],
    [255, 1, 100],
    [255, 1, 200],
    [1, 255, 1],
    [255, 1, 1],
    [250, 170, 29],
    [250, 170, 26],
    [250, 170, 25],
    [250, 170, 24],
    [250, 170, 22],
    [250, 170, 21],
    [250, 170, 20],
    [255, 255, 255],
    [250, 170, 19],
    [250, 170, 18],
    [250, 170, 12],
    [250, 170, 11],
    [255, 255, 255],
    [255, 255, 255],
    [250, 170, 16],
    [250, 170, 15],
    [250, 170, 15],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [64, 170, 64],
    [230, 160, 50],
    [70, 130, 180],
    [190, 255, 255],
    [152, 251, 152],
    [107, 142, 35],
    [1, 170, 30],
    [255, 255, 128],
    [250, 1, 30],
    [100, 140, 180],
    [220, 128, 128],
    [222, 40, 40],
    [100, 170, 30],
    [40, 40, 40],
    [33, 33, 33],
    [100, 128, 160],
    [20, 20, 255],
    [142, 1, 1],
    [70, 100, 150],
    [250, 171, 30],
    [250, 173, 30],
    [250, 174, 30],
    [250, 175, 30],
    [250, 176, 30],
    [210, 170, 100],
    [153, 153, 153],
    [128, 128, 128],
    [1, 1, 80],
    [210, 60, 60],
    [250, 170, 30],
    [250, 170, 30],
    [250, 170, 30],
    [250, 170, 30],
    [250, 170, 30],
    [250, 170, 30],
    [192, 192, 192],
    [192, 192, 192],
    [220, 220, 1],
    [220, 220, 1],
    [1, 1, 196],
    [192, 192, 192],
    [220, 220, 1],
    [140, 140, 20],
    [119, 11, 32],
    [150, 1, 255],
    [1, 60, 100],
    [1, 1, 142],
    [1, 1, 90],
    [1, 1, 230],
    [1, 80, 100],
    [128, 64, 64],
    [1, 1, 110],
    [1, 1, 70],
    [1, 1, 192],
    [170, 170, 170],
    [32, 32, 32],
    [111, 74, 1],
    [120, 10, 10],
    [81, 1, 81],
    [111, 111, 1]
])

MAPILLARY12 = np.array([
    [165, 42, 42],
    [0, 192, 0],
    [196, 196, 196],
    [190, 153, 153],
    [180, 165, 180],
    [90, 120, 150],
    [102, 102, 156],
    [128, 64, 255],
    [140, 140, 200],
    [170, 170, 170],
    [250, 170, 160],
    [96, 96, 96],
    [230, 150, 140],
    [128, 64, 128],
    [110, 110, 110],
    [244, 35, 232],
    [150, 100, 100],
    [70, 70, 70],
    [150, 120, 90],
    [220, 20, 60],
    [255, 0, 0],
    [255, 0, 100],
    [255, 0, 200],
    [200, 128, 128],
    [255, 255, 255],
    [64, 170, 64],
    [230, 160, 50],
    [70, 130, 180],
    [190, 255, 255],
    [152, 251, 152],
    [107, 142, 35],
    [0, 170, 30],
    [255, 255, 128],
    [250, 0, 30],
    [100, 140, 180],
    [220, 220, 220],
    [220, 128, 128],
    [222, 40, 40],
    [100, 170, 30],
    [40, 40, 40],
    [33, 33, 33],
    [100, 128, 160],
    [142, 0, 0],
    [70, 100, 150],
    [210, 170, 100],
    [153, 153, 153],
    [128, 128, 128],
    [0, 0, 80],
    [250, 170, 30],
    [192, 192, 192],
    [220, 220, 0],
    [140, 140, 20],
    [119, 11, 32],
    [150, 0, 255],
    [0, 60, 100],
    [0, 0, 142],
    [0, 0, 90],
    [0, 0, 230],
    [0, 80, 100],
    [128, 64, 64],
    [0, 0, 110],
    [0, 0, 70],
    [0, 0, 192],
    [32, 32, 32],
    [120, 10, 10],
    [0, 0, 0],
])

COCO_PANOPTIC = np.array([
    [220, 20, 60],
    [119, 11, 32],
    [0, 0, 142],
    [0, 0, 230],
    [106, 0, 228],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 70],
    [0, 0, 192],
    [250, 170, 30],
    [100, 170, 30],
    [220, 220, 0],
    [175, 116, 175],
    [250, 0, 30],
    [165, 42, 42],
    [255, 77, 255],
    [0, 226, 252],
    [182, 182, 255],
    [0, 82, 0],
    [120, 166, 157],
    [110, 76, 0],
    [174, 57, 255],
    [199, 100, 0],
    [72, 0, 118],
    [255, 179, 240],
    [0, 125, 92],
    [209, 0, 151],
    [188, 208, 182],
    [0, 220, 176],
    [255, 99, 164],
    [92, 0, 73],
    [133, 129, 255],
    [78, 180, 255],
    [0, 228, 0],
    [174, 255, 243],
    [45, 89, 255],
    [134, 134, 103],
    [145, 148, 174],
    [255, 208, 186],
    [197, 226, 255],
    [171, 134, 1],
    [109, 63, 54],
    [207, 138, 255],
    [151, 0, 95],
    [9, 80, 61],
    [84, 105, 51],
    [74, 65, 105],
    [166, 196, 102],
    [208, 195, 210],
    [255, 109, 65],
    [0, 143, 149],
    [179, 0, 194],
    [209, 99, 106],
    [5, 121, 0],
    [227, 255, 205],
    [147, 186, 208],
    [153, 69, 1],
    [3, 95, 161],
    [163, 255, 0],
    [119, 0, 170],
    [0, 182, 199],
    [0, 165, 120],
    [183, 130, 88],
    [95, 32, 0],
    [130, 114, 135],
    [110, 129, 133],
    [166, 74, 118],
    [219, 142, 185],
    [79, 210, 114],
    [178, 90, 62],
    [65, 70, 15],
    [127, 167, 115],
    [59, 105, 106],
    [142, 108, 45],
    [196, 172, 0],
    [95, 54, 80],
    [128, 76, 255],
    [201, 57, 1],
    [246, 0, 122],
    [191, 162, 208],
    [255, 255, 128],
    [147, 211, 203],
    [150, 100, 100],
    [168, 171, 172],
    [146, 112, 198],
    [210, 170, 100],
    [92, 136, 89],
    [218, 88, 184],
    [241, 129, 0],
    [217, 17, 255],
    [124, 74, 181],
    [70, 70, 70],
    [255, 228, 255],
    [154, 208, 0],
    [193, 0, 92],
    [76, 91, 113],
    [255, 180, 195],
    [106, 154, 176],
    [230, 150, 140],
    [60, 143, 255],
    [128, 64, 128],
    [92, 82, 55],
    [254, 212, 124],
    [73, 77, 174],
    [255, 160, 98],
    [255, 255, 255],
    [104, 84, 109],
    [169, 164, 131],
    [225, 199, 255],
    [137, 54, 74],
    [135, 158, 223],
    [7, 246, 231],
    [107, 255, 200],
    [58, 41, 149],
    [183, 121, 142],
    [255, 73, 97],
    [107, 142, 35],
    [190, 153, 153],
    [146, 139, 141],
    [70, 130, 180],
    [134, 199, 156],
    [209, 226, 140],
    [96, 36, 108],
    [96, 96, 96],
    [64, 170, 64],
    [152, 251, 152],
    [208, 229, 228],
    [206, 186, 171],
    [152, 161, 64],
    [116, 112, 0],
    [0, 114, 143],
    [102, 102, 156],
    [250, 141, 255],
    [0, 0, 0]
])

CULANE_NOC = 5
BDD100K_DRIVABLE_NOC = 3
BDD100K_LANE_CONVERT_NOC = 9
BDD100K_SEMANTIC_SEGMENTATION_NOC = 19
MAPILLARY_NOC = 116
MAPILLARY12_NOC = 65
CITYSCAPES_NOC = 19
COCO_PANOPTIC_NOC = 133
OPENLANE_LANE_TYPE_NOC = 15

CITYSCAPES_CLASS_NAMES = \
    ['road',
     'sidewalk',
     'building',
     'wall',
     'fence',
     'pole',
     'traffic light',
     'traffic sign',
     'vegetation',
     'terrain',
     'sky',
     'person',
     'rider',
     'car',
     'truck',
     'bus',
     'train',
     'motorcycle',
     'bicycle'
     ]

MAPILLARY12_CLASS_NAMES = \
    ['Bird',
     'Ground Animal',
     'Curb',
     'Fence',
     'Guard Rail',
     'Barrier',
     'Wall',
     'Bike Lane',
     'Crosswalk - Plain',
     'Curb Cut',
     'Parking',
     'Pedestrian Area',
     'Rail Track',
     'Road',
     'Service Lane',
     'Sidewalk',
     'Bridge',
     'Building',
     'Tunnel',
     'Person',
     'Bicyclist',
     'Motorcyclist',
     'Other Rider',
     'Lane Marking - Crosswalk',
     'Lane Marking - General',
     'Mountain',
     'Sand',
     'Sky',
     'Snow',
     'Terrain',
     'Vegetation',
     'Water',
     'Banner',
     'Bench',
     'Bike Rack',
     'Billboard',
     'Catch Basin',
     'CCTV Camera',
     'Fire Hydrant',
     'Junction Box',
     'Mailbox',
     'Manhole',
     'Phone Booth',
     'Pothole',
     'Street Light',
     'Pole',
     'Traffic Sign Frame',
     'Utility Pole',
     'Traffic Light',
     'Traffic Sign (Back)',
     'Traffic Sign (Front)',
     'Trash Can',
     'Bicycle',
     'Boat',
     'Bus',
     'Car',
     'Caravan',
     'Motorcycle',
     'On Rails',
     'Other Vehicle',
     'Trailer',
     'Truck',
     'Wheeled Slow',
     'Car Mount',
     'Ego Vehicle'
     ]

COCO_PANOPTIC_CLASS_NAMES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
    'banner',
    'blanket',
    'bridge',
    'cardboard',
    'counter',
    'curtain',
    'door-stuff',
    'floor-wood',
    'flower',
    'fruit',
    'gravel',
    'house',
    'light',
    'mirror-stuff',
    'net',
    'pillow',
    'platform',
    'playingfield',
    'railroad',
    'river',
    'road',
    'roof',
    'sand',
    'sea',
    'shelf',
    'snow',
    'stairs',
    'tent',
    'towel',
    'wall-brick',
    'wall-stone',
    'wall-tile',
    'wall-wood',
    'water-other',
    'window-blind',
    'window-other',
    'tree-merged',
    'fence-merged',
    'ceiling-merged',
    'sky-other-merged',
    'cabinet-merged',
    'table-merged',
    'floor-other-merged',
    'pavement-merged',
    'mountain-merged',
    'grass-merged',
    'dirt-merged',
    'paper-merged',
    'food-other-merged',
    'building-other-merged',
    'rock-merged',
    'wall-other-merged',
    'rug-merged'
]

CITYSCAPES_THING_LIST = [11, 12, 13, 14, 15, 16, 17, 18]
MAPILLARY12_THING_LIST = [0, 1, 8, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48,
                          49,
                          50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62]

COCO_PANOPTIC_THING_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                            19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48,
                            49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
                            73, 74, 75, 76, 77, 78, 79]


def get_color_palette_and_number_of_classes(dataset_name, dataset_target_type):
    if dataset_name == "Culane":
        color_palette = CULANE
        number_of_classes = CULANE_NOC

    elif dataset_name == "Mapillary":
        number_of_classes = MAPILLARY_NOC
        color_palette = MAPILLARY

    elif dataset_name == "Mapillary12":
        number_of_classes = MAPILLARY12_NOC
        color_palette = MAPILLARY12

    elif dataset_name == "BDD100k" or dataset_name == "BDD100kVideo":
        color_palette, number_of_classes = handle_multiple_head_data_structure(bdd_single_target_selector, dataset_target_type)

    elif dataset_name == "Openlane":
        color_palette, number_of_classes = handle_multiple_head_data_structure(openlane_target_selector, dataset_target_type)

    elif dataset_name == "Cityscapes":
        number_of_classes = CITYSCAPES_NOC
        color_palette = CITYSCAPES

    elif dataset_name == "CocoPanoptic":
        number_of_classes = COCO_PANOPTIC_NOC
        color_palette = COCO_PANOPTIC

    else:
        raise ValueError("Not a supported dataset...")

    return color_palette, number_of_classes


def handle_multiple_head_data_structure(function, dataset_target_type):
    if isinstance(dataset_target_type, list):
        color_palette = []
        number_of_classes = []
        for target_type in dataset_target_type:
            single_color_palette, single_number_of_class = function(target_type)
            color_palette.append(single_color_palette)
            number_of_classes.append(single_number_of_class)
    else:
        color_palette, number_of_classes = function(dataset_target_type)

    return color_palette, number_of_classes

def get_thing_list_and_class_names(dataset_name):
    if dataset_name == "Mapillary12":
        class_names = MAPILLARY12_CLASS_NAMES
        thing_list = MAPILLARY12_THING_LIST

    elif dataset_name == "Cityscapes":
        class_names = CITYSCAPES_CLASS_NAMES
        thing_list = CITYSCAPES_THING_LIST

    elif dataset_name == "CocoPanoptic":
        class_names = COCO_PANOPTIC_CLASS_NAMES
        thing_list = COCO_PANOPTIC_THING_LIST

    else:
        raise ValueError("Not a supported dataset...")

    return class_names, thing_list


def bdd_single_target_selector(target_type):
    if target_type == "drivable-masks":
        color_palette = BDD100K_DRIVABLE
        number_of_class = BDD100K_DRIVABLE_NOC

    elif target_type == "lane-bitmasks_morph":
        color_palette = BDD100K_LANE_CONVERT
        number_of_class = 3

    elif target_type == "sem_seg-masks":
        color_palette = BDD100K_SEMANTIC_SEGMENTATION
        number_of_class = BDD100K_SEMANTIC_SEGMENTATION_NOC

    elif target_type == "lane-deeplabv3_resnet18_culane_640x368":
        color_palette = CULANE
        number_of_class = CULANE_NOC

    elif target_type == "sem_seg-deeplabv3_resnet50_mapillary_1856x1024_apex":
        color_palette = MAPILLARY
        number_of_class = MAPILLARY_NOC

    elif target_type == "lane-simplified":
        color_palette = BDD100K_SINGLE_LANE
        number_of_class = 2

    else:
        raise ValueError("not a valid target for BDD100K dataset")

    return color_palette, number_of_class


def openlane_target_selector(target_type):
    if "semantic_culane" in target_type:
        color_palette = CULANE
        number_of_class = CULANE_NOC

    elif target_type == "semantic_lane_type":
        color_palette = OPENLANE_LANE_TYPE
        number_of_class = OPENLANE_LANE_TYPE_NOC

    elif target_type == "semantic_binary":
        color_palette = BDD100K_SINGLE_LANE
        number_of_class = 2

    else:
        raise ValueError("not a valid target for OpenLane dataset")

    return color_palette, number_of_class

