import torch
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms


class XRayPipelineBaseline:
    """ Implement basic transformations for XRay image dataset 
    """
    def __init__(self, size):
        assert len(size) == 2
        self.size = size

    def common(self):
        """ Common transformation pipeline.
        
        __call__: 
            .. Input expect PIL RGB PIL Image of shape: (H,W,C)
            .. Output return torch.Tensor Image of shape (C,H,W)
        """
        pipelines = transforms.Compose([
            transforms.CenterCrop(size=self.size),
            transforms.ToTensor(),
            transforms.Normalize()
        ])
        return pipelines

    def augmentation(self):
        """ Common transformation pipeline.

        __call__: 
            .. Input expect PIL.Image of shape: (H,W,C)
            .. Output return PIL.Image of shape (H,W,C)
        """
        return transforms.Compose([])


class XRayPipelineCLAHE:
    """ Implement xray transformations with CLAHE processing
    """
    pass