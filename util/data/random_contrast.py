# Author: Kyle Ma @ BCIL 
# Created: 05/05/2023
# Implementation of Automated Hematoma Segmentation

import torch

def random_contrast(image, random, wsize):

    # set treshold according to the paper
    low_threshold = 0
    high_threshold = 140

    # default weight is 1
    weight = torch.tensor([1.0])

    # add random contrast
    if random:
        low_value = torch.distributions.uniform.Uniform(-wsize,wsize).sample([1])
        high_value = torch.distributions.uniform.Uniform(-wsize,wsize).sample([1])

        low_threshold += low_value
        high_threshold += high_value

        # calculate the weight according to paper
        weight -= (torch.abs(high_value) + torch.abs(low_value))/ (2 * wsize)

    # create masks
    low_mask = image < low_threshold
    high_mask = image > high_threshold
    mid_mask = (image >= low_threshold) & (image <= high_threshold)

    # modify the dicom images
    image[low_mask] = 0
    image[mid_mask] = (image[mid_mask] - low_threshold) / (high_threshold - low_threshold) * 255
    image[high_mask] = 255

    return image, weight