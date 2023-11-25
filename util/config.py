# Author: Kyle Ma @ BCIL 
# Created: 05/12/2023
# Implementation of Automated Hematoma Segmentation


class Config():

    # Experiment Congigurations
    epoch_number = 5
    batch_size = 2
    learning_rate = 0.001
    eval_ratio = 0.1
    
    model_level = 4
    fix_split = True
    model_type = "Unet"
    # Unet, RRUnet, ResUnet, DRUnet, AttUnet, AttRRUnet
    # MV, MV_SE, MV_SE_RES, MV_F, MV_eval

    # Test the new data or not
    test_new = True
    img_ct = True

    # Data Loading Configurations
    load_num = 5
    training_directory = "#####"
    new_data_directory = "#####"

    # Experiment Output Folder Name
    exp_series = 'testpipeline'

    if model_type[0] == "M":
        if fix_split:
            exp_name = exp_series +'_locksplit_' + model_type + '_level' + str(model_level) + "_epoch" + str(epoch_number) + "_lr" + str(learning_rate)
        else:
            exp_name = exp_series + model_type + '_level' + str(model_level) + "_epoch" + str(epoch_number) + "_lr" + str(learning_rate)
    else:
        if fix_split:
            exp_name = exp_series +'_locksplit_' + model_type  + "_epoch" + str(epoch_number) + "_lr" + str(learning_rate)
        else:
            exp_name = exp_series + model_type + "_epoch" + str(epoch_number) + "_lr" + str(learning_rate)