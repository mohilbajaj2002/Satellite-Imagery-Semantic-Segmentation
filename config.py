import os
import sys

# File to enter basic data about the project

project_name = 'satellite_imagery'

classes = {'Water': (226, 169, 41),
           'Land': (132, 41, 246),
           'Road': (110, 193, 228),
           'Building': (60, 16, 152),
           'Vegetation': (254, 221, 58),
           'Unlabeled': (155, 155, 155)}

# Folders & Paths
raw_data_folder = 'data'
aggregate_data_folder = 'agg_data'
saved_model_folder = 'saved_model'
saved_history_folder = 'saved_history'
saved_history_plots_folder = 'saved_history_plots'

root_path = os.path.dirname(os.path.realpath(sys.argv[0]))
data_root_path = os.path.join(root_path, raw_data_folder)
saved_model_root_path = os.path.join(root_path, saved_model_folder)
saved_history_root_path = os.path.join(root_path, saved_history_folder)
saved_history_plots_path = os.path.join(root_path, saved_history_plots_folder)

# For data pre-processing and aumentation
seed = 1  # for repeatability and image/mask matching
train_test_split = 0.8
augmentation_multiplier_list = [8] # 8, 15, 25, 50, 100 etc.

# Model parameters & other training options
num_classes = len(classes.keys())
lr_scheduler_epoch_factor = 0.2
early_stopping_epoch_factor = 0.75
# input shape list format = [(resnet152_shape[0], resnet152_shape[1], other_backbone_shape[0], other_backbone_shape[1])]
input_shape_list = [(256, 256, 400, 400)] # Resnet152  = (256, 256), (400, 400), (512, 512) etc.; All others = (400, 400), (512, 512), etc.
depth_selector_list = ['3'] # '3', '4', '5'
backbone_list = ['Resnet50', 'Resnet152', 'InceptionResNet'] # 'Resnet50', 'Resnet101', 'Resnet152', 'InceptionResNet'
batch_size_list = [16] # 8, 16, 32, 64, 128
epoch_list = [1] # 50, 100, 200
optimizer_list = ['SGD']  # 'SGD', 'RMSProp', 'Adam'
learning_rate_list = [0.0001] # 0.0001, 0.0005, 0.001, 0.0015 etc.
multiprocessing_toggle = False

# for prediction
best_model = ''
prediction_image_path = ''

# For data visualization
#viz_type = 'raw_vs_augmented_data' # 'raw_data', 'raw_vs_augmented_data', 'model_training_history', 'predictions'
#viz_model_name = ''
