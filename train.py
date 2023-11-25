# Author: Kyle Ma @ BCIL 
# Created: 04/26/2023
# Implementation of Automated Hematoma Segmentation

import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Customized Files
from util.config import Config
from util.test_new import test_new
from util.evaluate import evaluate
from util.loss import SoftDiceLoss
from util.data.augment import DataAug
from util.data.loader import load_data
from util.models.network import get_model
from util.data.split import train_val_split
from util.data.random_contrast import random_contrast



def run_model(model, device, dir_checkpoint, dir_exp):

    # 1. Read data from matlab file
    data, patient_condition = load_data(config.load_num, config.training_directory)

    # 2. Train eval Split
    train_set, val_set = train_val_split(data, patient_condition, config.eval_ratio, config.fix_split)
    number_train = len(train_set)
    
    # perform data augmentation (elastic transform, horizontal flip)
    train_set = DataAug(train_set)

    # create data loader
    train_loader_args = dict(drop_last = True, shuffle = False, batch_size = config.batch_size, 
                       num_workers = 0, pin_memory = True)
    train_loader = DataLoader(train_set, **train_loader_args)

    # The original paper uses adam with learning rate of 0.001
    optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)
    
    # proposed mixed loss
    criterion = SoftDiceLoss()

    # logging the training configuration
    logging.info(f'''Starting training:
        Model:           {model.__class__.__name__}
        Epochs:          {config.epoch_number}
        Batch size:      {config.batch_size}
        Learning rate:   {config.learning_rate}
        Training size:   {number_train}
    ''')

    # remember how many step total has been taken
    global_step = 0

    max_training_dice = 0
    max_testing_dice = 0
    max_training_dice_checkpoint = None
    max_testing_dice_checkpooint = None

    loss_list = []
    validation_dice_list = []
    training_dice_list = []

    # train for 200 epochs
    for epoch in range(1, config.epoch_number+1):
        
        running_loss = 0
        running_dice = 0
        
        # start the training
        model.train()

        with tqdm(total = number_train, desc = f'Epoch {epoch}/{config.epoch_number}', unit = ' img') as pbar:

            # for every batch of images
            for i, (images, masks) in enumerate(train_loader):

                # random contrast
                images, weight = random_contrast(images, False, 30)
                
                # move the images to gpu or cpu
                images = images.to(device)
                masks = masks.to(device)
                weight = weight.to(device)
                
                # add zero grad
                optimizer.zero_grad()

                # get the prediction
                prediction = model(images.float())

                # calculate loss
                mix_loss = criterion(prediction, masks, weight)
                dice_loss = criterion(prediction, masks, torch.tensor([1.0]).to(device))
                
                # gradient descent
                mix_loss.backward()                
                optimizer.step()
                global_step += 1
                
                # update the tqdm pbar
                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss (batch)': mix_loss.item()})
                
                # calculating average loss dice
                running_loss += mix_loss.item()
                running_dice += 1 - dice_loss.item()





        # print loss
        logging.info('average train loss is {} ; dice is {} ; step {} ; epoch {}.'.format(running_loss / len(train_loader), running_dice / len(train_loader), global_step, epoch))

        # get evaluation score
        val_score = evaluate(model, val_set, device)

        # for plotting validation dice
        validation_dice_list.append(val_score)
        # for plotting dice and loss
        loss_list.append(running_loss / len(train_loader))
        training_dice_list.append(running_dice / len(train_loader))

        # log the evaluation score
        logging.info(f'Validation Average Subject Dice Score is {val_score}')


        # saving the best performed ones
        if (running_dice / len(train_loader)) >= max_training_dice:
            max_training_dice_checkpoint = model.state_dict().copy()
            max_training_dice = running_dice / len(train_loader)
        if val_score >= max_testing_dice:
            max_testing_dice = val_score
            max_testing_dice_checkpooint = model.state_dict().copy()


    # save the max checkpoints
    torch.save(max_training_dice_checkpoint, os.path.join(dir_checkpoint,'Max_Training_Dice_{}.pt'.format(max_training_dice)))
    torch.save(max_testing_dice_checkpooint, os.path.join(dir_checkpoint,'Max_Testing_Dice_{}.pt'.format(max_testing_dice)))
    # save the model after job is done
    torch.save(model.state_dict(), os.path.join(dir_checkpoint,'COMPLETED.pt'))


    logging.info('Training Completed Model Saved')
    logging.info('Max Training Dice is {}'.format(max_training_dice))
    logging.info('Max Testing Dice is {}'.format(max_testing_dice))

    x = np.linspace(1, len(training_dice_list), len(training_dice_list))
    plt.figure()
    plt.plot(x, training_dice_list)
    plt.xlabel('Steps')
    plt.ylabel('Dice Score')
    plt.title('Training Dice')
    plt.savefig('{}/TrainingDice.png'.format(dir_exp))
    plt.close()

    plt.figure()
    plt.plot(x, loss_list)
    plt.xlabel('Steps')
    plt.ylabel('MIxed Loss')
    plt.title('Training Loss')
    plt.savefig('{}/TrainingLoss.png'.format(dir_exp))
    plt.close()


    y = np.linspace(1, len(validation_dice_list), len(validation_dice_list))
    plt.figure()
    plt.plot(y, validation_dice_list)
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.title('Testing Dice')
    plt.savefig('{}/TestingDice.png'.format(dir_exp))
    plt.close()

    if config.test_new:
        test_new(model, device, dir_exp, config)


if __name__ == '__main__':

    config = Config()
    
    # create output directory
    dir_output = './outputs/'
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)
    
    # create output directory
    dir_exp = './outputs/{}'.format(config.exp_name)
    if not os.path.isdir(dir_exp):
        os.mkdir(dir_exp)
    

    # initialize the logging
    logging.basicConfig(filename='{}/Running.log'.format(dir_exp), level=logging.INFO, format='%(asctime)s: %(message)s')
    
    # use GPU if it is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log which device we are using
    logging.info(f"Model is Running on {device}.")


    model = get_model(config.model_type)


    # move the model to gpu or cpu
    model.to(device)


    # create check point directory
    dir_checkpoint = '{}/checkpoints/'.format(dir_exp)
    if not os.path.isdir(dir_checkpoint):
        os.mkdir(dir_checkpoint)

    # run the model and save interrupt
    try:
        run_model(model, device, dir_checkpoint, dir_exp)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(dir_checkpoint,'INTERRUPTED.pt'))
        logging.info("Saved Interrupt at INTERRUPTED.pt")
