import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from torchcraft.utils import helper
import torchvision.transforms as transforms

class CIFAR10Albumentation:
    
    def __init__(self):
        pass
    
    def train_transform(self,mean,std):
        # Train Phase transformations
        train_transforms = A.Compose([A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                                      A.RandomCrop(width=32, height=32,p=1),
                                      A.Rotate(limit=5),
                                      #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.25),
                                      A.CoarseDropout(max_holes=1,min_holes = 1, max_height=16, max_width=16, p=0.5,fill_value=tuple([x * 255.0 for x in mean]),
                                      min_height=16, min_width=16),
                                      A.Normalize(mean=mean, std=std,always_apply=True),
                                      ToTensorV2()
                                    ])
        return lambda img:train_transforms(image=np.array(img))["image"]
                                
    def test_transform(self,mean,std):
        # Test Phase transformations
        test_transforms = A.Compose([A.Normalize(mean=mean, std=std, always_apply=True),
                                 ToTensorV2()])
        return lambda img:test_transforms(image=np.array(img))["image"]
        
        
class CIFAR10AlbumentationS9:
    
    def __init__(self):
        pass
    
    def train_transform(self,mean,std):
        # Train Phase transformations
        train_transforms = A.Compose([A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                                      A.RandomCrop(width=32, height=32,p=1),
                                      A.HorizontalFlip(p=1),
                                      A.CoarseDropout(max_holes=3,min_holes = 1, max_height=8, max_width=8, p=0.8,fill_value=tuple([x * 255.0 for x in mean]),
                                      min_height=8, min_width=8),
                                      A.Normalize(mean=mean, std=std,always_apply=True),
                                      ToTensorV2()
                                    ])
        return lambda img:train_transforms(image=np.array(img))["image"]
                                
    def test_transform(self,mean,std):
        # Test Phase transformations
        test_transforms = A.Compose([A.Normalize(mean=mean, std=std, always_apply=True),
                                 ToTensorV2()])
        return lambda img:test_transforms(image=np.array(img))["image"]
        
        
class TinyImageNetAlbumentation:
    
    def __init__(self):
        pass
    
    def train_transform(self,mean,std):
        # Train Phase transformations
        train_transforms = A.Compose([  A.PadIfNeeded(min_height=70, min_width=70, always_apply=True),
                                        A.RandomCrop(height=64, width=64,p=1),
                                        A.Rotate(limit=5),
                                        A.HorizontalFlip(p=0.5),
                                        A.CoarseDropout(max_holes=1,min_holes = 1, max_height=32, max_width=32, p=0.8,fill_value=tuple([x * 255.0 for x in mean]),
                                      min_height=32, min_width=32),
                                        A.Normalize(mean=mean, std=std,always_apply=True),
                                        ToTensorV2()
                                    ])
        return lambda img:train_transforms(image=np.array(img))["image"]
                                
    def test_transform(self,mean,std):
        # Test Phase transformations
        test_transforms = A.Compose([A.Normalize(mean=mean, std=std, always_apply=True),
                                 ToTensorV2()])
        return lambda img:test_transforms(image=np.array(img))["image"]
