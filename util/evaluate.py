# Author: Kyle Ma @ BCIL 
# Created: 04/28/2023
# Implementation of Automated Hematoma Segmentation

import torch
import numpy as np
from util.loss import SoftDiceLoss
from torch.utils.data import DataLoader
from util.data.random_contrast import random_contrast


def evaluate(model, val_set, device):
    # start the evaluation mode
    model.eval()

    scorer = SoftDiceLoss()

    # contain total scores
    dice_score = 0

    for patient in val_set:

        # create test loader for each patients
        test_loader_args = dict(drop_last = False, shuffle = False, batch_size = 1, 
                   num_workers = 0, pin_memory = True)
        test_loader = DataLoader(patient, **test_loader_args)

        # store mask and prediction per patients
        predictions = []
        masks = []
    
        # for every slices in the test loader
        for i, (images, mask, condition) in enumerate(test_loader):

            #make it 4D with batch size = 1
            images = images.unsqueeze(1)

            images, weight = random_contrast(images, False, 30)
            
            # move the images to gpu or cpu
            images = images.to(device)
            weight = weight.to(device)

            # get our prediction
            with torch.no_grad():
                prediction = model(images.float())

                predictions.append(prediction.cpu())
                masks.append(mask.cpu())
    
        dice_score += (1 - scorer(torch.stack(predictions), torch.stack(masks), 1).item())   

        
    model.train()
    
    return dice_score / len(val_set)



            
            
    
    
    
    
    



