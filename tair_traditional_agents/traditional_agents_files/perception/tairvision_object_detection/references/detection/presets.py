from tairvision.references.detection.transforms import *
import tairvision.references.detection.transforms as T


class DetectionPresetTrain:
    def __init__(self, data_augmentation, is_train=True, min_size=800, max_size=1333, hflip_prob=0.5,
                 fill_mean=(123.0, 117.0, 104.0), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.data_augmentation = data_augmentation
        if data_augmentation == 'hflip':
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(p=hflip_prob),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
                T.ResizeWithMask(skip_resize=False, is_train=is_train, min_size=min_size, max_size=max_size)
            ])
        elif data_augmentation == 'retinaface':
            self.transforms = T.Compose([
                T.RandomPhotometricDistort(),
                T.RandomCrop(),
                T.Resize((640, 640)),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
                T.ResizeWithMask(skip_resize=False, is_train=is_train, min_size=min_size, max_size=max_size)
            ])
        elif data_augmentation == 'distort':
            self.transforms = T.Compose([
                T.RandomPhotometricDistort(),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
                T.ResizeWithMask(skip_resize=False, is_train=is_train, min_size=min_size, max_size=max_size)
            ])
        elif data_augmentation == 'weather':
            try:
                import albumentations as A
            except ImportError:
                raise ImportError("no module named albumentations")

            t = []
            transforms_a = A.Compose([
                A.RandomBrightnessContrast(p=hflip_prob),
                A.RandomRain(blur_value=4),
            ])
            transforms_t = T.Compose([
                T.RandomPhotometricDistort(),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
                T.ResizeWithMask(skip_resize=False, is_train=is_train, min_size=min_size, max_size=max_size)
            ])

            t.append(transforms_a)
            t.append(transforms_t)
            self.transforms = T.CustomCompose(t)

        elif data_augmentation == 'lsj':
            self.transforms = T.Compose([
                T.ScaleJitter(target_size=(1024, 1024)),
                T.FixedSizeCrop(size=(1024, 1024), fill=fill_mean),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
                T.ResizeWithMask(skip_resize=True, is_train=is_train, min_size=min_size, max_size=max_size)
            ])
        elif data_augmentation == "multiscale":
            self.transforms = T.Compose(
                [
                    T.RandomShortestSize(
                        min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333
                    ),
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.PILToTensor(),
                    T.ConvertImageDtype(torch.float),
                    T.Normalize(mean=mean, std=std),
                    T.ResizeWithMask(skip_resize=True, is_train=is_train, min_size=min_size, max_size=max_size)
                ]
            )
        elif data_augmentation == "multiscale_box_normalized":
            self.transforms = T.Compose(
                [
                    T.RandomShortestSize(
                        min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333
                    ),
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.PILToTensor(),
                    T.ConvertImageDtype(torch.float),
                    T.NormalizeAlsoBoxes(mean=mean, std=std)
                ]
            )

        elif data_augmentation == "multiscale_box_normalized_openlaneV2":
            self.transforms = T.Compose(
                [
                    T.RandomShortestSize(
                        min_size=[960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600], max_size=2080
                    ),
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.PILToTensor(),
                    T.ConvertImageDtype(torch.float),
                    T.NormalizeAlsoBoxes(mean=mean, std=std)
                ]
            )

        elif data_augmentation == "multiscale_box_normalized_detr":
            scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
            self.transforms = T.Compose(
                [
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.RandomSelect(
                        T.RandomShortestSize(
                            min_size=scales, max_size=1333
                        ),
                        T.Compose([
                            T.RandomShortestSize([400, 500, 600], max_size=5000),
                            T.RandomSizeCrop(384, 600),
                            T.RandomShortestSize(scales, max_size=1333),
                        ])
                    ),
                    T.PILToTensor(),
                    T.ConvertImageDtype(torch.float),
                    T.NormalizeAlsoBoxes(mean=mean, std=std)
                ]
            )
        elif data_augmentation is None:
            self.transforms = T.Compose([
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
                T.ResizeWithMask(skip_resize=False, is_train=is_train, min_size=min_size, max_size=max_size)
            ])
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self, data_augmentation, min_size=800, max_size=1333, is_train=False, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        if data_augmentation == 'retinaface':
            self.transforms = T.Compose([
                T.RandomCrop(),
                T.Resize((640, 640)),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
                T.ResizeWithMask(is_train=is_train, min_size=min_size, max_size=max_size),
            ])
        elif data_augmentation == 'demo':
            self.transforms = T.Compose([
                T.Resize((640, 640)),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
                T.ResizeWithMask(is_train=is_train, min_size=min_size, max_size=max_size),
            ])
        elif data_augmentation == 'standard_normalized':
            self.transforms = T.Compose([
                T.RandomShortestSize(
                    min_size=800, max_size=1333
                ),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.NormalizeAlsoBoxes(mean=mean, std=std),
            ])
        elif data_augmentation == 'standard_normalized_openlaneV2':
            self.transforms = T.Compose([
                T.RandomShortestSize(
                    min_size=1600, max_size=2080
                ),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.NormalizeAlsoBoxes(mean=mean, std=std),
            ])
        else:
            self.transforms = T.Compose([
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
                T.ResizeWithMask(is_train=is_train, min_size=min_size, max_size=max_size),
            ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(is_train, data_aug, min_size, max_size):
    return DetectionPresetTrain(data_augmentation=data_aug, is_train=is_train, min_size=min_size, max_size=max_size) \
        if is_train else DetectionPresetEval(data_augmentation=data_aug, is_train=is_train, min_size=min_size,
                                             max_size=max_size)
