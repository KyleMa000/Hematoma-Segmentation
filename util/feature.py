# Author: Kyle Ma @ BCIL 
# Created: 05/25/2023
# Implementation of Automated Hematoma Segmentation

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.loader import LoadMatData
from data.random_contrast import random_contrast
from util.models.experiment.mv_eval import MV_eval


def featureMap(checkpoint_path, level, data, patient_num, slice_num):

    # load the model and checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MV_eval(level)
    path = checkpoint_path
    model.to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    # get the slice we interested in
    image = torch.tensor(data[patient_num-1][slice_num][0])
    image = image.unsqueeze(0)
    image = image.unsqueeze(0)

    # perform random contrast
    image, weight = random_contrast(image, False, 30)
    image = image.to(device)

    # run the image through the model
    with torch.no_grad():
        prediction, Qs, Ps = model(image.float())

    # process Q feature to gray scale
    processed_Q = []
    for Q in Qs:
        Q = Q.squeeze(0)
        gray_scale = torch.sum(Q,0)
        gray_scale = gray_scale / Q.shape[0]
        processed_Q.append(gray_scale.data.cpu().numpy())

    # process P feature to gray scale
    processed_P = []
    for P in Ps:
        P = P.squeeze(0)
        gray_scale = torch.sum(P,0)
        gray_scale = gray_scale / P.shape[0]
        processed_P.append(gray_scale.data.cpu().numpy())

    for i in range(len(processed_Q)):
        plt.imshow(processed_Q[i])
        plt.title("Q{}".format(i))
        plt.show()

    for i in range(len(processed_P)):
        plt.imshow(processed_P[i])
        plt.title("P{}".format(len(processed_P) - i - 1))
        plt.show()
        
    prediction = torch.sigmoid(prediction)
    
    mask = data[patient_num-1][slice_num][1]

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(mask, cmap='Greys')
    ax1.set_title('GroundTruth')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.imshow(prediction.cpu()[0][0], cmap='Greys')
    ax2.set_title('Prediction')
    ax2.set_xticks([])
    ax2.set_yticks([])