# Author: Kyle Ma @ BCIL 
# Created: 04/26/2023
# Implementation of Automated Hematoma Segmentation

from torch import nn
import torch

# The standard diceloss for testing
def dice_coeff(pred, target):
    smooth = 0.00001
    num = pred.size(0)
    
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets, weight):

        # get probability with sigmoid function
        probs = torch.sigmoid(logits)

        score = dice_coeff(probs, targets) * weight

        return 1 - score