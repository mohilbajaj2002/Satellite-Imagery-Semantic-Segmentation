# Satellite Imagery Semantic Segmentation

This project aims to detect features like land, water, road, buildings etc. in satellite imagery using 2D Semantic Segmentation. The dataset used for the project is an open-source dataset initially published by Humans In the Loop and is available at this link: https://humansintheloop.org/resources/datasets/semantic-segmentation-dataset-2/. Model architectures like U-Net with varying depths (3, 4 and 5) and backbones like Resnet50, Resnet101, Resnet152 and Inception-ResnetV2 were explored. Data pre-processing steps included image resizing and normalization. Data Augmentation was done using horizontal & vertical flipping, random cropping, image contrast enhancement & brightness rectification, rotation and optical & grid distortion.

The project consists of 8 files and 1 folder:
- Data : This is the folder where the downloaded zip file should be extracted.
- Requirements.txt : This file lists all the libraries needed for this project.
- Config.py : This file contains all the project variables and parameters.
- Architectures.py : This file contains different model architectures discussed above.
- Data_generation.py : This file contains functions related to the train and validation generator.
- Utilities.py : This file contains code for performing performance evaluation, graph plotting and enhancements related to code readability.
- Main.py : This is the main file. It performs data pre-processing and model training.
- Visualization.py : This file can be used to perform data exploration, if needed.
- Prediction.py : This is the file for making prediction.

The project has been designed to be a one-shot solution for model training and evaluation (after your preferences have been added to the config file) and thus does not require human intervention or interaction once initiated. This makes it most suited for High Performance Computing (HPC) environments. For environments like single GPU/Google Colab, the main file will need to be modified to break processing time into smaller chunks. An additional visualization file is also added for standalone data exploration or to compare data before and after augmentation.

While most models performed reasonably well, U-Net with depth 4, Inception-ResnetV2 backbone, image size of (400, 400) and a data augmentation multiplier of 25 performed the best with a Dice-Coefficient of 0.83 when averaged over 5 trials. Techniques like deeper U-Net designs, larger data augmentation multiplier and image sizes, and hyper-parameter tuning could be used to further improve performance though this may require clusters with higher memory capabilities and bigger GPUs.

To initiate, download the data to the data folder and add your preferences to the config.py file and then run main.py. Once model(s) have been trained, update the prediction section in the config file and use the appropriate file to make prediction(s).

PS: More architectures to be added later.
