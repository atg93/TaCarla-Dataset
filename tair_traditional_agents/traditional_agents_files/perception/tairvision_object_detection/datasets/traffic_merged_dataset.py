from torch.utils.data import Dataset
from PIL import Image

class edaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir

        self.transform = transform

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, index):
        image_filepath = self.root_dir[index]

        label = image_filepath.split('/')[-1]
        label = label.split('_')[0]
        label = int(label)

        image = Image.open(image_filepath)

        if self.transform is not None:
            img = self.transform(image)

        return img, label