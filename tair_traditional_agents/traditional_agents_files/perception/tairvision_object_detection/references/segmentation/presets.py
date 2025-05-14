from tairvision.references.segmentation.transforms import *
import tairvision.references.segmentation.transforms as T


class SegmentationPresetTrain:
    def __init__(self, data_augmentation=None, base_size=None, crop_size=None, hflip_prob=0.5, resize_ratio = [0.5, 2.0],
                 jitter_aspect_ratio = [0.75, 1.33], rotation = [-10, 10],
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.data_augmentation = data_augmentation
        self.crop_size = crop_size
        self.base_size = base_size
        if self.crop_size is None:
            raise ValueError("Crop size cannot be None!!!")

        if self.base_size is None:
            self.base_size = self.crop_size

        self.resize_ratio = resize_ratio
        self.jitter_aspect_ratio = jitter_aspect_ratio
        self.rotation = rotation
        self.resize_range = [self.base_size * resize_ratio[0], self.base_size * resize_ratio[1]]
        if self.data_augmentation is None:
            min_size = int(self.resize_ratio[0] * self.base_size)
            max_size = int(self.resize_ratio[1] * self.base_size)

            trans = [T.RandomResize(min_size, max_size)]
            if hflip_prob > 0:
                trans.append(T.RandomHorizontalFlip(hflip_prob))
            trans.extend([
                T.RandomCrop(crop_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
            self.transforms = T.Compose(trans)
        elif self.data_augmentation == "culane_lane_fit_specific":
            trans = [T.RandomHorizontalFlip(modify_label=True, number_of_class=5),
                     T.RandomResize(288, 560),
                     T.RandomCropWithRandomPadding(360, 640),
                     T.ToTensor(),
                     T.Normalize(mean=mean, std=std)]
            self.transforms = T.Compose(trans)
        elif self.data_augmentation == "culane_strong_data_aug":
            trans = [
                # T.RandomHorizontalFlip(modify_label=True, number_of_class=5),
                T.RandomPhotometricDistort(),
                T.RandomRotation(degrees=self.rotation, fill=125),
                T.RandomPerspective(p=0.5, fill=125),
                T.JitterAspectRatio(ratio=self.jitter_aspect_ratio),
                T.RandomResize(min_size=self.resize_range[0], max_size=self.resize_range[1]),
                T.RandomCropWithRandomPadding(size=(self.crop_size, self.crop_size), image_pad_intensity=125),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ]
            self.transforms = T.Compose(trans)
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, data_augmentation=None, base_size=None, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.data_augmentation = data_augmentation
        if self.data_augmentation is None:
            self.transforms = T.Compose([
                T.RandomResize(base_size, base_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetNoShaping:
    def __init__(self):
        self.transforms = T.Compose([
            T.ToTensor(),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)
