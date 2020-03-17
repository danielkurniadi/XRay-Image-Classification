import os
from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms

from .xray_pipeline import XRayPipelineBaseline


def abs_listdir(root):
    return [os.path.join(root, dir)
            for dir in os.listdir()]


def get_filepaths_labels(file_root):
    filepaths = []
    labels = []

    for class_folder in abs_listdir(file_root):
        filepaths += abs_listdir(class_folder)
        
        label = PnemoniaXRayDataset.LABEL_ENCODING[class_folder.upper()]
        labels += label * len(filepaths)

    return filepaths, labels


class PnemoniaXRayDataset(Dataset):
    """ Dataset class for Pnemonia XRay Image 
    """

    LABEL_ENCODING = {
        'PNEMONIA': 1,
        'NORMAL': 0
    }


    def __init__(self, file_root, size, train=True):
        """ Load pnemonia images from files, given the directory to the root.
        
        Args:
            .. file_root (str): path to data directory
            .. size (tuple): height, width of the image of each data
            .. train (bool): whether this dataset is for train (True) or test/val (False)
        """
        assert os.path.isdir(file_root) "File directory of dataset is not found: {}".format(fileroot)

        self.file_root = file_root
        self.train = train
        
        # first recursively find all datasets + labels in the file root
        # this assumes each data is grouped by folder based on the class target/label
        self.filepaths, self.labels = get_filepaths_labels(file_root)

        assert len(self.filepaths) == len(self.labels)
        
        # Transformation class
        self.pipeline = XRayPipelineBaseline(size)
        self.transforms = self.define_transforms()

    def define_transforms(self):
        """ Define transformations from our transform pipeline
        """
        if self.train == True:
            return transforms.Compose([self.pipeline.augment(), self.pipeline.common()])
        else:
            return self.pipeline.common()

    # --------------------
    # Generator related
    # --------------------
    def __len__(self):
        return len(self.filepaths)
        
    def __get_item__(self, index):
        filename = self.filenames[index]
        label = self.labels[index]

        image = Image.open(filename)
        image = self.transforms(image)
        return image, label
