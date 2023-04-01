import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

import config
import utilities as utils
import architectures as arch
import data_generation as datagen


# Create project artifact folder
os.makedirs(saved_prediction_plots_path, exist_ok=True)

# Gather model parameters
prediction_model = config.prediction_model

opt_name = prediction_model.split('_')[-5]
learning_rate = float(prediction_model.split('_')[-3])
optimizer = get_optimizer(opt_name, learning_rate)

depth = prediction_model.split('_')[3]
bb_name = prediction_model.split('_')[4]
model_input_shape = (int(prediction_model.split('_')[-8]), int(prediction_model.split('_')[-8]), 3)

# Build model and load weights
model = build_model(depth, bb_name, model_input_shape, config.num_classes, optimizer)
model_path = os.path.join(saved_model_root_path, prediction_model)
model.load_weights(model_path)

# Create prediction data generators
prediction_image_size = (int(prediction_model.split('_')[-8]), int(prediction_model.split('_')[-8]))
prediction_generator = AugmentDataGenerator(config.prediction_image_dir_path, config.prediction_mask_dir_path,
                                            seed = config.seed, batch_size = config.prediction_batch_size, target_size = prediction_image_size)

# Make predictions
prediction_plot_path = config.saved_prediction_plots_path
utils.make_prediction(model, prediction_generator, prediction_plot_path)
