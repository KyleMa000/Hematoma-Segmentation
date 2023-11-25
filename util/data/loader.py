# Author: Kyle Ma @ BCIL 
# Created: 05/16/2023
# Implementation of Automated Hematoma Segmentation

import os
import h5py
import torch
import logging
import numpy as np
from tqdm import tqdm

# save data into .pt format to reduce loading time
def load_data(num, directory):
    pt_path = directory + "data.pt"
    if os.path.isfile(pt_path):
        logging.info(f"Loading Data From: {pt_path}")
        data = torch.load(pt_path)
        return data[0], data[1]
    else:
        return load_mat(num, directory)
    

def load_mat(num, directory):
    # container for patient data
    data = []
    patient_condition = []
    patient_number = 0

    # There are 5 files of .mat training data
    for i in range(1,num+1):

        # reading the file
        path = directory + "PatientsData_{}.mat".format(i)

        # log which data we are loading
        logging.info(f"Loading Data From: {path}")

        f = h5py.File(path, 'r')

        # get the number of patients
        patient_index = range(f['PatientsData']['dicomImgs'].shape[0])

        # for every patient
        for patient in tqdm(patient_index):

            patient_number += 1

            # get input image and annotation references
            image_ref = f['PatientsData']['dicomImgs'][patient][0]
            annotation_ref = f['PatientsData']['annots'][patient][0]

            # the number of frame inthe image
            slices_num = f[image_ref].shape[0]

            patient_data = []

            Healthy = True

            for slice in range(slices_num):

                # get image and annotation and change datatype
                image = f[image_ref][slice,:,:]
                annotation = f[annotation_ref][slice,:,:]

                if np.sum(annotation) > 0:
                    patient_data.append([image, annotation, 1])
                    Healthy = False
                else:
                    patient_data.append([image, annotation, 0])    

            patient_condition.append(Healthy)
            data.append(patient_data)

        f.close()
        
    logging.info(f"Loaded {len(data)} Data")

    pt_path = directory + "data.pt"

    torch.save((data, patient_condition), pt_path)

    logging.info(f"Data saved to {pt_path}")

    return data, patient_condition