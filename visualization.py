import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

import config
import utilities as utils

# Create project artifact folder
os.makedirs(config.misc_data_exploration_path, exist_ok=True)

# Visualize data
if(config.viz_type = 'raw_data'):
    utils.raw_data_viz(config.original_image_path, config.original_mask_path,
                       config.misc_data_exploration_path) 
elif(config.viz_type = 'raw_vs_augmented_data'):
    utils.raw_vs_transformed_data_viz(config.transformed_image_path, config.transformed_mask_path,
                                      config.original_image_path, config.original_mask_path,
                                      config.misc_data_exploration_path)
