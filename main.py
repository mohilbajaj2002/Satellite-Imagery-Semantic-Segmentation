import pickle
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import matplotlib.pyplot as plt
import os, re, sys, random, shutil, cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import model_to_dot, plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import InceptionResNetV2, ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, LearningRateScheduler
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, ZeroPadding2D, Dropout

import config
import utilities as utils
import architectures as arch
import data_generation as datagen


# Creating Model Artifact folders
os.makedirs(config.saved_model_root_path, exist_ok=True)
os.makedirs(config.saved_history_root_path, exist_ok=True)
os.makedirs(config.saved_history_plots_path, exist_ok=True)

# Aggregating data in one place
datagen.aggregate_data(config.data_root_path, config.aggregate_data_folder)

# Augmenting data and splitting it into train and validation datasets
for aug_multiplier in augmentation_multiplier_list:
    print(f'Creating new data with multiplier effect of {aug_multiplier} ...')
    utils.create_augmented_dataset(config.data_root_path, config.aggregate_data_folder, aug_multiplier)
    utils.split_dataset_into_train_and_val(config.data_root_path, config.train_test_split, aug_multiplier)
    train_image_folder = os.path.join(config.data_root_path, f'aug_data_{aug_multiplier}', 'train', 'images')
    val_image_folder = os.path.join(config.data_root_path, f'aug_data_{aug_multiplier}', 'val', 'images')
    train_mask_folder = os.path.join(config.data_root_path, f'aug_data_{aug_multiplier}', 'train', 'masks')
    val_mask_folder = os.path.join(config.data_root_path, f'aug_data_{aug_multiplier}', 'val', 'masks')
    # Starting model training
    for batch_size in config.batch_size_list:
        for input_shape in config.input_shape_list:
            for depth in config.depth_selector_list:
                for bb_name in config.backbone_list:
                    final_input_shape = utils.get_input_shape(bb_name, input_shape)
                    train_generator = AugmentDataGenerator(train_image_folder, train_mask_folder, seed = config.seed, batch_size = batch_size, target_size = final_input_shape)
                    validation_generator = AugmentDataGenerator(val_image_folder, val_mask_folder, seed = config.seed, batch_size = batch_size, target_size = final_input_shape)
                    print('Train and Validation generators created successfully!')
                    for opt_name in config.optimizer_list:
                        for learning_rate in config.learning_rate_list:
                            optimizer = utils.get_optimizer(opt_name, learning_rate)
                            for epochs in config.epoch_list:
                                num_classes = config.num_classes
                                model_no = len(os.listdir(config.saved_model_root_path)) + 1
                                datatype = aug_multiplier
                                model_input_shape = (final_input_shape[0], final_input_shape[1], 3)
                                model_name =  f'Attempt_{model_no}_UNet_{depth}_{bb_name}_augtype_{datatype}_imagesize_{final_input_shape[0]}_batchsize_{batch_size}_{opt_name}_learningrate_{learning_rate}_epochs_{epochs}'
                                model_path = os.path.join(config.saved_model_root_path, model_name)
                                history_path = os.path.join(config.saved_history_root_path, f'{model_name}.csv')
                                model = arch.build_model(depth, bb_name, model_input_shape, num_classes, optimizer)
                                steps_per_epoch, validation_steps = datagen.calculate_number_of_samples(config.data_root_path, train_image_folder, val_image_folder, batch_size)
                                model, history = arch.train_model(model, epochs, train_generator, validation_generator, steps_per_epoch, validation_steps, model_path, history_path, config.multiprocessing_toggle)

                                # Model Performance
                                utils.plot_history(history, model_name, True)
                                utils.display_performance_metrics(history, model_name)
