import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torch import from_numpy
from skimage.color import rgb2lab


class ImageDataset(Dataset):
    def __init__(self, color_dir, gray_dir = None, transform = None, target_transform = None):
        """
        :param color_dir: The directory where the colored images are located at.
        :param gray_dir: (Optional) The directory where the gray image are located at.
        When this parameter is not set, LAB format is used. Otherwise, RGB is used.
        :param transform: (Optional) `transform` function to be applied on a gray image.
        :param target_transform: (Optional) `transform` function to be applied on a colored image.
        """
        self.names = os.listdir(color_dir)[:4]
        self.color_dir = color_dir
        self.gray_dir = gray_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Returns the length of the dataset.
        :return: The length of the dataset.
        """
        return len(self.names)

    def __getitem__(self, index):
        """
        Fetches the item of the given index.
        :param index: The index where the item is located at.
        :return: The item in the given index.
        """
        if self.gray_dir is not None:
            gray_path = os.path.join(self.gray_dir, self.names[index])
            gray_image = read_image(gray_path, ImageReadMode.GRAY)

            color_path = os.path.join(self.color_dir, self.names[index])
            color_image = read_image(color_path)
        else:
            color_path = os.path.join(self.color_dir, self.names[index])
            image = from_numpy(rgb2lab(read_image(color_path).permute(1, 2, 0))).permute(2, 0, 1)

            # The color image consists of the 'a' and 'b' parts of the LAB format.
            color_image = image[1:, :, :]
            # The gray image consists of the `L` part of the LAB format.
            gray_image = image[0, :, :].unsqueeze(0)

        if self.transform:
            gray_image = self.transform(gray_image)
        if self.target_transform:
            color_image = self.target_transform(color_image)

        return gray_image, color_image
