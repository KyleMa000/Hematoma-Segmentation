# Author: Kyle Ma @ BCIL 
# Created: 05/02/2023
# Implementation of Automated Hematoma Segmentation

import random
import logging
import numpy as np

def train_val_split(data, patient_condition, eval_ratio, fix_split):

    if fix_split:
        random.seed(727)

    data = np.array(data, dtype=object)

    # get the number of evaluation and training
    number_val = int(len(data) * eval_ratio)
    number_train = len(data) - number_val

    # evaluation set only keeps unhealthy patients
    val_temp = data[np.logical_not(patient_condition)]
    # training set can have healthy patients
    train_temp = data[patient_condition]

    # get validation set random patient
    random.shuffle(val_temp)
    val_set = val_temp[:number_val]

    # get back our train set with healthy patients
    train_set = np.concatenate((val_temp[number_val:],train_temp))

    # separate positive and negative data for training
    positive_data = []
    negative_data = []

    for patient in train_set:
        for slices in patient:
            if slices[-1] == 1:
                positive_data.append(slices[:2])
            else:
                negative_data.append(slices[:2])

    # shuffle to random order
    random.shuffle(positive_data)
    random.shuffle(negative_data)

    # data balancing method
    mix_data = []
    for i in range(len(positive_data)):
        mix_data.append(positive_data[i])
        mix_data.append(negative_data[i])
        
    logging.info(f"There are {len(mix_data)} Training Data")
    logging.info(f"There are {len(val_set)} Patient for Validation")
        
    return mix_data, val_set