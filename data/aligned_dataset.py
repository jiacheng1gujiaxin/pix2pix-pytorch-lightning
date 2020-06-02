import os
from torch.utils.data import Dataset
from data.base_dataset import get_params, get_transform
from PIL import Image

from util.str2label import get_convert


class AlignedDataset(Dataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, phase, load_size, crop_size, preprocess):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            dataroot (str) -- images dir
            phase (str) -- train, val, test
            load_size (tuple) -- (width, height) if None don`t resize
            crop_size (tuple) -- (width, height) if None don`t crop
            preprocess (list) -- ['resize', 'crop', 'flip']
        """
        super(AlignedDataset, self).__init__()
        self.dir_AB = os.path.join(dataroot, phase)  # get the image directory
        self.AB_paths = sorted(os.listdir(self.dir_AB)) # get image paths
        self.phase = phase
        self.load_size = load_size
        self.crop_size = crop_size
        self.preprocess = preprocess
        self.convert = get_convert()
        if load_size is not None and crop_size is not None:
            assert(load_size[0] >= crop_size[0] and load_size[1] >= crop_size[1])   # crop_size should be smaller than the size of loaded image


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        # text to tensor -- name (idx_text.jpg)

        AB = Image.open(os.path.join(self.dir_AB, AB_path)).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        tranform = get_transform(self.preprocess, self.load_size, self.crop_size,
                                 grayscale=False, method=Image.BICUBIC, convert=True)
        if self.phase != 'test':
            A = tranform(A)
            B = tranform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
