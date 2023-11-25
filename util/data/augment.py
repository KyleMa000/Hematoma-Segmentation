# Author: Kyle Ma @ BCIL 
# Created: 05/01/2023
# Implementation of Automated Hematoma Segmentation

import torch
import random
import torchvision.transforms.functional as TF
from torch.utils.data import TensorDataset as Dataset
from util.data.elastic_transform import do_elastic_transform

class DataAug(Dataset):
    def __init__(self, data):
        self.data = data

    def transform(self, dataset):
        
        # Transform a copy to Tensor so original does not change
        image = TF.to_tensor(dataset[0].copy())
        mask = TF.to_tensor(dataset[1].copy())

        # Random Elastic Transformation
        if random.random() > 0.5:

            # do elastic transform to the image and mask at same time
            data = torch.cat((image, mask), dim = 0)
            data = do_elastic_transform(data)

            # separate them and turn back to original shape
            image = data[0]
            mask = data[1]

            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)

            # turn back the masks to binary
            mask[mask > 0] = 1
            mask[mask != 1] = 0

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
                    
        return image, mask

    def __getitem__(self, index):

        dataset = self.data[index]

        # apply the transformation
        image, mask = self.transform(dataset)


        return image, mask

    def __len__(self):
        return len(self.data)